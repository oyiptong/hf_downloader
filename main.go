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
	"github.com/rodaine/table"    // Import the new table package
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
	force     bool

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
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		var err error
		db, err = initDB(dbName)
		if err != nil {
			log.Printf("Error initializing database: %v\n", err)
			return fmt.Errorf("failed to initialize database: %w", err)
		}
		return nil
	},
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
	Args: cobra.ArbitraryArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		if inputFile == "" && len(args) == 0 {
			return fmt.Errorf("you must provide URLs either via the --file flag or as arguments")
		}
		if inputFile != "" && len(args) > 0 {
			return fmt.Errorf("you cannot use both the --file flag and provide URLs as arguments simultaneously")
		}
		if outputDir == "" {
			return fmt.Errorf("the --output-dir flag is required")
		}

		outputDir = filepath.Clean(outputDir)
		fmt.Printf("Using output directory: %s\n", outputDir)
		if force {
			fmt.Println("Force mode enabled: Existing entries will be updated.")
		}

		var urlsToProcess []string
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
				if line != "" && !strings.HasPrefix(line, "#") {
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
			urlsToProcess = args
			fmt.Printf("Processing %d URLs from arguments.\n", len(urlsToProcess))
		}

		tx, err := db.Begin()
		if err != nil {
			return fmt.Errorf("failed to begin database transaction: %w", err)
		}
		defer tx.Rollback()

		var sqlCmd string
		if force {
			sqlCmd = `
                INSERT INTO ` + tableName + ` (url, downloaded, date_added, destination_path)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    downloaded = excluded.downloaded,
                    date_added = excluded.date_added,
                    destination_path = excluded.destination_path;
            `
		} else {
			sqlCmd = "INSERT INTO " + tableName + "(url, downloaded, date_added, destination_path) VALUES(?, ?, ?, ?)"
		}

		stmt, err := tx.Prepare(sqlCmd)
		if err != nil {
			return fmt.Errorf("failed to prepare SQL statement: %w", err)
		}
		defer stmt.Close()

		processedCount := 0
		skippedInvalidCount := 0
		skippedDuplicateCount := 0

		for _, rawURL := range urlsToProcess {
			fmt.Printf("Processing URL: %s\n", rawURL)
			author, filename, err := parseHuggingFaceURL(rawURL)
			if err != nil {
				fmt.Printf("  Skipping (Invalid URL/Format) - %v\n", err)
				skippedInvalidCount++
				continue
			}

			destinationPath := filepath.Join(outputDir, author, filename)
			currentTime := time.Now().Format(time.RFC3339)

			result, err := stmt.Exec(rawURL, false, currentTime, destinationPath)
			if err != nil {
				if !force {
					var sqliteErr sqlite3.Error
					if errors.As(err, &sqliteErr) && sqliteErr.Code == sqlite3.ErrConstraint && sqliteErr.ExtendedCode == sqlite3.ErrConstraintUnique {
						fmt.Printf("  Skipping (Duplicate URL)\n")
						skippedDuplicateCount++
						continue
					}
				}
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
		return nil
	},
}

// listCmd represents the list command
var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List URLs from the database in a table format",
	Long: `Lists URLs stored in the database, displaying date_added, url, and destination_path.
By default, it shows URLs that have not yet been downloaded (downloaded = false).
Use the --downloaded flag to list URLs that have been marked as downloaded.
URLs are ordered by the date they were added in ascending order.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		var querySuffix string
		var queryParams []interface{}

		if listDownloaded {
			fmt.Println("Listing downloaded URLs (ordered by date added):")
			querySuffix = " WHERE downloaded = ? ORDER BY date_added ASC"
			queryParams = append(queryParams, true)
		} else {
			fmt.Println("Listing not downloaded URLs (ordered by date added):")
			querySuffix = " WHERE downloaded = ? ORDER BY date_added ASC"
			queryParams = append(queryParams, false)
		}

		query := "SELECT date_added, url, destination_path FROM " + tableName + querySuffix

		rows, err := db.Query(query, queryParams...)
		if err != nil {
			return fmt.Errorf("failed to query database: %w", err)
		}
		defer rows.Close()

		tbl := table.New("Date Added", "URL", "Destination Path")
		tbl.WithHeaderFormatter(func(format string, vals ...interface{}) string {
			return strings.ToUpper(fmt.Sprintf(format, vals...))
		})
		tbl.WithPadding(2)              // Adjust padding as needed
		tbl.WithHeaderSeparatorRow('-') // Use '=' as header separator

		found := false
		for rows.Next() {
			found = true
			var dateAdded, itemUrl, destinationPath string
			if err := rows.Scan(&dateAdded, &itemUrl, &destinationPath); err != nil {
				log.Printf("Error scanning row: %v", err)
				continue
			}
			// Attempt to parse and reformat date for better readability if needed
			parsedTime, err := time.Parse(time.RFC3339, dateAdded)
			displayDate := dateAdded // Default to original string if parsing fails
			if err == nil {
				displayDate = parsedTime.Format("2006-01-02 15:04:05") // Example format
			}
			tbl.AddRow(displayDate, itemUrl, destinationPath)
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
		} else {
			tbl.Print()
		}
		return nil
	},
}

// init initializes the cobra command structure
func init() {
	rootCmd.AddCommand(loadCmd)
	rootCmd.AddCommand(listCmd)

	loadCmd.Flags().StringVarP(&inputFile, "file", "f", "", "Path to a text file containing URLs (one per line)")
	loadCmd.Flags().StringVarP(&outputDir, "output-dir", "o", "", "Directory to store downloaded files (required)")
	loadCmd.Flags().BoolVar(&force, "force", false, "Overwrite existing URL entry, updating date_added, destination_path, and resetting downloaded status")
	if err := loadCmd.MarkFlagRequired("output-dir"); err != nil {
		log.Fatalf("Failed to mark 'output-dir' flag as required for load command: %v", err)
	}

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
	filename = parts[len(parts)-1]
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
