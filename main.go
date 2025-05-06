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

			// If we reach here, the exec was successful (either insert or update)
			rowsAffected, _ := result.RowsAffected() // Check if rows were affected (usually 1 for insert/update)

			// In force mode, ON CONFLICT DO UPDATE might affect 1 row (the updated one).
			// A standard INSERT affects 1 row.
			// We need a way to know if it was an *update*.
			// A simple way without extra queries: If force is true and rowsAffected is 1,
			// assume it could have been an update or a new insert.
			// A more precise way would be to query first, but ON CONFLICT is generally preferred.
			// Let's refine the logic: If force=true, we consider it "processed" regardless of insert/update.
			// We can try to infer if it was an update by checking if the ID already existed *before* the transaction,
			// but that adds complexity. Let's stick to simpler logging for now.

			if force && rowsAffected > 0 { // rowsAffected should be 1 for insert or update
				// We can't easily distinguish insert vs update with ON CONFLICT without another query
				// Let's count it as processed/updated when force is true
				fmt.Printf("  Added or Updated in database. Destination: %s\n", destinationPath)
				// Let's increment a general 'processed' count in force mode for simplicity
				processedCount++ // Or potentially updatedCount++ if we could reliably detect updates
			} else if !force && rowsAffected > 0 { // Standard insert successful
				fmt.Printf("  Added to database. Destination: %s\n", destinationPath)
				processedCount++
			} else if rowsAffected == 0 && force {
				// This *shouldn't* typically happen with ON CONFLICT DO UPDATE unless the data was identical,
				// but we log it just in case.
				fmt.Printf("  Entry already exists with identical data (no changes made).\n")
				// Treat as skipped duplicate for counting purposes in force mode? Or processed?
				// Let's count as processed as the intent was to ensure it exists.
				processedCount++
			}
			// Note: If rowsAffected is 0 in non-force mode, an error should have occurred (caught above).

		}

		// Commit the transaction if all operations were successful
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit database transaction: %w", err)
		}

		fmt.Printf("\nProcessing complete.\n")
		if force {
			// In force mode, "processed" includes both new inserts and updates.
			fmt.Printf("  Processed (Added or Updated): %d\n", processedCount)
		} else {
			fmt.Printf("  Successfully added:           %d\n", processedCount)
			fmt.Printf("  Skipped (Duplicate):          %d\n", skippedDuplicateCount)
		}
		fmt.Printf("  Skipped (Invalid URL):        %d\n", skippedInvalidCount)

		return nil // Success
	},
}

// init initializes the cobra command structure
func init() {
	// Add loadCmd as a subcommand to rootCmd
	rootCmd.AddCommand(loadCmd)

	// Define flags for loadCmd
	loadCmd.Flags().StringVarP(&inputFile, "file", "f", "", "Path to a text file containing URLs (one per line)")
	loadCmd.Flags().StringVarP(&outputDir, "output-dir", "o", "", "Directory to store downloaded files (required)")
	loadCmd.Flags().BoolVar(&force, "force", false, "Overwrite existing URL entry, updating date_added, destination_path, and resetting downloaded status") // Add force flag

	// Mark outputDir as required
	if err := loadCmd.MarkFlagRequired("output-dir"); err != nil {
		log.Fatalf("Failed to mark 'output-dir' flag as required: %v", err)
	}

}

// initDB initializes the SQLite database connection and creates the table if it doesn't exist.
func initDB(dbPath string) (*sql.DB, error) {
	// Open the database file. It will be created if it doesn't exist.
	d, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL") // WAL mode often better for concurrency
	if err != nil {
		return nil, fmt.Errorf("could not open database %s: %w", dbPath, err)
	}

	// Check the connection
	if err = d.Ping(); err != nil {
		d.Close() // Close if ping fails
		return nil, fmt.Errorf("could not connect to database %s: %w", dbPath, err)
	}

	// SQL statement to create the table if it doesn't exist
	createTableSQL := `
    CREATE TABLE IF NOT EXISTS ` + tableName + ` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "url" TEXT UNIQUE NOT NULL,
        "downloaded" BOOLEAN NOT NULL DEFAULT 0,
        "date_added" TEXT NOT NULL,
        "destination_path" TEXT NOT NULL
    );`

	// Execute the SQL statement
	_, err = d.Exec(createTableSQL)
	if err != nil {
		d.Close() // Close if table creation fails
		return nil, fmt.Errorf("could not create table '%s': %w", tableName, err)
	}

	// fmt.Printf("Database '%s' initialized successfully.\n", dbPath) // Reduce verbosity
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
	filename = parts[len(parts)-1]

	if author == "" || filename == "" {
		return "", "", fmt.Errorf("could not extract non-empty author ('%s') or filename ('%s') from path '%s'", author, filename, u.Path)
	}

	return author, filename, nil
}

// main is the entry point of the application
func main() {
	// Execute command and handle potential errors
	if err := rootCmd.Execute(); err != nil {
		// Cobra usually prints the error, but ensure non-zero exit code
		// fmt.Fprintf(os.Stderr, "Error: %v\n", err) // Keep commented unless needed
		os.Exit(1)
	}
}
