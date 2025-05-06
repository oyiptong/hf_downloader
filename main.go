package main

import (
	"bufio"
	"database/sql"
	"errors" // Import errors package
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mattn/go-sqlite3" // Import the driver directly to check error types
	"github.com/spf13/cobra"
)

const (
	dbName          = "downloads.db"
	tableName       = "downloads"
	huggingFaceHost = "huggingface.co"
)

var (
	// Flags for the load command
	inputFile string
	outputDir string
	force     bool // New flag

	// Flag for the list command
	listDownloaded bool

	// Database connection
	db *sql.DB
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "hfdownloader",
	Short: "A CLI tool to manage Hugging Face model downloads",
	Long: `hfdownloader helps prepare download information for models hosted on Hugging Face.
It can parse URLs, store metadata in a database, and prepare destination paths.`,
	// PersistentPreRunE ensures the database is initialized before any command runs
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		var err error
		db, err = initDB(dbName)
		if err != nil {
			// Log initialization errors before cobra potentially suppresses them
			log.Printf("Error initializing database: %v\n", err)
			return fmt.Errorf("failed to initialize database: %w", err)
		}
		return nil
	},
	// PersistentPostRun closes the database connection after command execution
	PersistentPostRun: func(cmd *cobra.Command, args []string) {
		if db != nil {
			if err := db.Close(); err != nil {
				log.Printf("Error closing database: %v\n", err)
			}
		}
	},
}

// loadCmd represents the load command
var loadCmd = &cobra.Command{
	Use:   "load [URL...]",
	Short: "Load Hugging Face model URLs into the database",
	Long: `Loads Hugging Face model URLs either from a file specified with --file
or directly as space-separated arguments. It parses the URLs, determines
the author and filename, constructs a destination path based on the
--output-dir, and stores this information in the 'downloads.db' SQLite database.

By default, duplicate URLs are skipped. Use the --force flag to overwrite
existing entries, updating the date_added, destination_path, and resetting
the downloaded status to false.`,
	Args: cobra.ArbitraryArgs, // Allows URLs as arguments
	RunE: func(cmd *cobra.Command, args []string) error {
		// --- Input Validation ---
		if inputFile == "" && len(args) == 0 {
			return fmt.Errorf("you must provide URLs either via the --file flag or as arguments")
		}
		if inputFile != "" && len(args) > 0 {
			return fmt.Errorf("you cannot use both the --file flag and provide URLs as arguments simultaneously")
		}
		if outputDir == "" {
			return fmt.Errorf("the --output-dir flag is required")
		}

		// Clean the output directory path
		outputDir = filepath.Clean(outputDir)
		fmt.Printf("Using output directory: %s\n", outputDir)
		if force {
			fmt.Println("Force mode enabled: Existing entries will be updated.")
		}

		// --- URL Processing ---
		var urlsToProcess []string

		// Read URLs from file if specified
		if inputFile != "" {
			fmt.Printf("Reading URLs from file: %s\n", inputFile)
			file, err := os.Open(inputFile)
			if err != nil {
				return fmt.Errorf("failed to open input file '%s': %w", inputFile, err)
			}
			defer file.Close()

			scanner := bufio.NewScanner(file)
			for scanner.Scan() {
				line := strings.TrimSpace(scanner.Text())
				if line != "" && !strings.HasPrefix(line, "#") { // Ignore empty lines and comments
					urlsToProcess = append(urlsToProcess, line)
				}
			}
			if err := scanner.Err(); err != nil {
				return fmt.Errorf("error reading input file '%s': %w", inputFile, err)
			}
			if len(urlsToProcess) == 0 {
				fmt.Println("Warning: Input file was empty or contained no valid URLs.")
				return nil
			}
		} else {
			// Use URLs from arguments
			urlsToProcess = args
			fmt.Printf("Processing %d URLs from arguments.\n", len(urlsToProcess))
		}

		// --- Database Insertion ---
		tx, err := db.Begin()
		if err != nil {
			return fmt.Errorf("failed to begin database transaction: %w", err)
		}
		// Defer rollback, commit will override this if successful
		defer tx.Rollback() // Use defer for cleanup

		// Prepare statement based on the force flag
		var sqlCmd string
		if force {
			// If the URL exists (conflict on 'url'), update specified fields
			sqlCmd = `
                INSERT INTO ` + tableName + ` (url, downloaded, date_added, destination_path)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    downloaded = excluded.downloaded,
                    date_added = excluded.date_added,
                    destination_path = excluded.destination_path;
            `
		} else {
			// Standard insert, duplicates will cause an error handled below
			sqlCmd = "INSERT INTO " + tableName + "(url, downloaded, date_added, destination_path) VALUES(?, ?, ?, ?)"
		}

		stmt, err := tx.Prepare(sqlCmd)
		if err != nil {
			// Rollback is handled by defer
			return fmt.Errorf("failed to prepare SQL statement: %w", err)
		}
		defer stmt.Close() // Ensure statement is closed

		processedCount := 0
		skippedInvalidCount := 0
		skippedDuplicateCount := 0

		for _, rawURL := range urlsToProcess {
			fmt.Printf("Processing URL: %s\n", rawURL)

			author, filename, err := parseHuggingFaceURL(rawURL)
			if err != nil {
				fmt.Printf("  Skipping (Invalid URL/Format) - %v\n", err)
				skippedInvalidCount++
				continue // Skip this URL
			}

			// Construct destination path
			destinationPath := filepath.Join(outputDir, author, filename)

			// Get current time in ISO 8601 format
			currentTime := time.Now().Format(time.RFC3339)

			// Execute the prepared statement
			// Always set downloaded to false when adding/forcing
			result, err := stmt.Exec(rawURL, false, currentTime, destinationPath)
			if err != nil {
				// Only check for constraint errors if not in force mode
				if !force {
					var sqliteErr sqlite3.Error
					// Check if the error is a SQLite error and specifically a UNIQUE constraint violation
					if errors.As(err, &sqliteErr) && sqliteErr.Code == sqlite3.ErrConstraint && sqliteErr.ExtendedCode == sqlite3.ErrConstraintUnique {
						fmt.Printf("  Skipping (Duplicate URL)\n")
						skippedDuplicateCount++
						continue // Skip this URL, it already exists
					}
				}
				// If it's another error (or a constraint error in force mode, which shouldn't happen with ON CONFLICT),
				// rollback and return the error. Rollback is handled by defer.
				return fmt.Errorf("failed to execute statement for URL '%s': %w", rawURL, err)
			}

			rowsAffected, _ := result.RowsAffected()

			if force && rowsAffected > 0 {
				fmt.Printf("  Added or Updated in database. Destination: %s\n", destinationPath)
				processedCount++
			} else if !force && rowsAffected > 0 {
				fmt.Printf("  Added to database. Destination: %s\n", destinationPath)
				processedCount++
			} else if rowsAffected == 0 && force {
				fmt.Printf("  Entry already exists with identical data (no changes made).\n")
				processedCount++
			}
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit database transaction: %w", err)
		}

		fmt.Printf("\nProcessing complete.\n")
		if force {
			fmt.Printf("  Processed (Added or Updated): %d\n", processedCount)
		} else {
			fmt.Printf("  Successfully added:           %d\n", processedCount)
			fmt.Printf("  Skipped (Duplicate):          %d\n", skippedDuplicateCount)
		}
		fmt.Printf("  Skipped (Invalid URL):        %d\n", skippedInvalidCount)

		return nil // Success
	},
}

// listCmd represents the list command
var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List URLs from the database",
	Long: `Lists URLs stored in the database.
By default, it shows URLs that have not yet been downloaded (downloaded = false).
Use the --downloaded flag to list URLs that have been marked as downloaded.
URLs are ordered by the date they were added in ascending order.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		var query string
		var queryParams []interface{}

		if listDownloaded {
			fmt.Println("Listing downloaded URLs (ordered by date added):")
			query = "SELECT url, date_added, destination_path FROM " + tableName + " WHERE downloaded = ? ORDER BY date_added ASC"
			queryParams = append(queryParams, true)
		} else {
			fmt.Println("Listing not downloaded URLs (ordered by date added):")
			query = "SELECT url, date_added, destination_path FROM " + tableName + " WHERE downloaded = ? ORDER BY date_added ASC"
			queryParams = append(queryParams, false)
		}

		rows, err := db.Query(query, queryParams...)
		if err != nil {
			return fmt.Errorf("failed to query database: %w", err)
		}
		defer rows.Close()

		found := false
		for rows.Next() {
			found = true
			var itemUrl, dateAdded, destinationPath string
			if err := rows.Scan(&itemUrl, &dateAdded, &destinationPath); err != nil {
				log.Printf("Error scanning row: %v", err) // Log and continue if possible
				continue
			}
			fmt.Printf("  URL: %s\n    Added: %s\n    Destination: %s\n", itemUrl, dateAdded, destinationPath)
		}

		if err := rows.Err(); err != nil {
			return fmt.Errorf("error iterating rows: %w", err)
		}

		if !found {
			if listDownloaded {
				fmt.Println("  No downloaded URLs found.")
			} else {
				fmt.Println("  No URLs pending download found.")
			}
		}
		return nil
	},
}

// init initializes the cobra command structure
func init() {
	// Add loadCmd as a subcommand to rootCmd
	rootCmd.AddCommand(loadCmd)
	rootCmd.AddCommand(listCmd) // Add the new listCmd

	// Define flags for loadCmd
	loadCmd.Flags().StringVarP(&inputFile, "file", "f", "", "Path to a text file containing URLs (one per line)")
	loadCmd.Flags().StringVarP(&outputDir, "output-dir", "o", "", "Directory to store downloaded files (required)")
	loadCmd.Flags().BoolVar(&force, "force", false, "Overwrite existing URL entry, updating date_added, destination_path, and resetting downloaded status")

	// Mark outputDir as required for loadCmd
	if err := loadCmd.MarkFlagRequired("output-dir"); err != nil {
		log.Fatalf("Failed to mark 'output-dir' flag as required for load command: %v", err)
	}

	// Define flags for listCmd
	listCmd.Flags().BoolVar(&listDownloaded, "downloaded", false, "List URLs that have been downloaded")
}

// initDB initializes the SQLite database connection and creates the table if it doesn't exist.
func initDB(dbPath string) (*sql.DB, error) {
	d, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL")
	if err != nil {
		return nil, fmt.Errorf("could not open database %s: %w", dbPath, err)
	}

	if err = d.Ping(); err != nil {
		d.Close()
		return nil, fmt.Errorf("could not connect to database %s: %w", dbPath, err)
	}

	createTableSQL := `
    CREATE TABLE IF NOT EXISTS ` + tableName + ` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "url" TEXT UNIQUE NOT NULL,
        "downloaded" BOOLEAN NOT NULL DEFAULT 0,
        "date_added" TEXT NOT NULL,
        "destination_path" TEXT NOT NULL
    );`

	_, err = d.Exec(createTableSQL)
	if err != nil {
		d.Close()
		return nil, fmt.Errorf("could not create table '%s': %w", tableName, err)
	}
	return d, nil
}

// parseHuggingFaceURL extracts the author and filename from a Hugging Face model URL.
func parseHuggingFaceURL(rawURL string) (author string, filename string, err error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", "", fmt.Errorf("invalid URL format: %w", err)
	}

	if u.Scheme != "https" || u.Host != huggingFaceHost {
		return "", "", fmt.Errorf("not a valid https://huggingface.co URL")
	}

	parts := strings.Split(strings.Trim(u.Path, "/"), "/")
	if len(parts) < 2 {
		return "", "", fmt.Errorf("URL path '%s' does not seem to contain author and filename", u.Path)
	}

	author = parts[0]
	filename = parts[len(parts)-1] // The last part is the filename

	// If the URL is like /<author>/<modelname>/resolve/<branch>/<filename>
	// The modelname might be parts[1] and filename parts[len(parts)-1]
	// If the URL is just /<author>/<filename_implying_modelname>
	// Then parts[0] is author, parts[1] is filename

	// Let's refine: The filename is always the last part.
	// The author is always the first part.
	// The path to the model might be outputDir/author/filename (if filename implies model)
	// OR outputDir/author/modelname/filename (if path is author/modelname/.../filename)
	// The current destination path construction `filepath.Join(outputDir, author, filename)`
	// seems to assume the filename is unique enough or implies the model.

	if author == "" || filename == "" {
		return "", "", fmt.Errorf("could not extract non-empty author ('%s') or filename ('%s') from path '%s'", author, filename, u.Path)
	}

	return author, filename, nil
}

// main is the entry point of the application
func main() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}
