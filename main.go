package main

import (
	"bufio"
	"bytes" // Added for capturing stderr
	"database/sql"
	"errors"
	"fmt"
	"log"
	"net/url"
	"os"
	"os/exec" // Added for running shell commands
	"path/filepath"
	"strings"
	"time"

	"github.com/mattn/go-sqlite3"
	"github.com/rodaine/table"
	"github.com/spf13/cobra"
)

const (
	dbName             = "downloads.db"
	downloadsTableName = "downloads"     // Renamed for clarity
	downloadRunsTable  = "download_runs" // New table name
	huggingFaceHost    = "huggingface.co"
)

var (
	// Flags for the load command
	inputFile string
	outputDir string
	force     bool

	// Flag for the list command
	listDownloaded bool

	// Flag for the download command
	downloadRetries int

	// Database connection
	db *sql.DB
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "hfdownloader",
	Short: "A CLI tool to manage Hugging Face model downloads",
	Long: `hfdownloader helps prepare download information for models hosted on Hugging Face.
It can parse URLs, store metadata in a database, prepare destination paths,
and download the files.`,
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

		// Ensure outputDir is an absolute path for consistent destination_path storage
		absOutputDir, err := filepath.Abs(outputDir)
		if err != nil {
			return fmt.Errorf("failed to get absolute path for output directory '%s': %w", outputDir, err)
		}
		outputDir = absOutputDir // Use the absolute path

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
		defer tx.Rollback() // Rollback if commit is not called

		var sqlCmd string
		if force {
			sqlCmd = `
                INSERT INTO ` + downloadsTableName + ` (url, downloaded, date_added, destination_path)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    downloaded = excluded.downloaded,
                    date_added = excluded.date_added,
                    destination_path = excluded.destination_path;
            `
		} else {
			sqlCmd = "INSERT INTO " + downloadsTableName + "(url, downloaded, date_added, destination_path) VALUES(?, ?, ?, ?)"
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
				// This case means the existing entry was identical to what we tried to insert/update.
				// For ON CONFLICT DO UPDATE, if no columns actually change, RowsAffected can be 0 for some drivers/versions.
				// For clarity, we can treat it as processed if no error occurred.
				fmt.Printf("  Entry already exists and was targeted for update (no effective change or re-inserted).\n")
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

		query := "SELECT id, date_added, url, destination_path FROM " + downloadsTableName + querySuffix

		rows, err := db.Query(query, queryParams...)
		if err != nil {
			return fmt.Errorf("failed to query database: %w", err)
		}
		defer rows.Close()

		tbl := table.New("ID", "Date Added", "URL", "Destination Path")
		tbl.WithHeaderFormatter(func(format string, vals ...interface{}) string {
			return strings.ToUpper(fmt.Sprintf(format, vals...))
		})
		tbl.WithPadding(2)
		tbl.WithHeaderSeparatorRow('-')

		found := false
		for rows.Next() {
			found = true
			var id int64
			var dateAdded, itemUrl, destinationPath string
			if err := rows.Scan(&id, &dateAdded, &itemUrl, &destinationPath); err != nil {
				log.Printf("Error scanning row: %v", err)
				continue
			}
			parsedTime, err := time.Parse(time.RFC3339, dateAdded)
			displayDate := dateAdded
			if err == nil {
				displayDate = parsedTime.Format("2006-01-02 15:04:05")
			}
			tbl.AddRow(id, displayDate, itemUrl, destinationPath)
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

// downloadCmd represents the download command
var downloadCmd = &cobra.Command{
	Use:   "download",
	Short: "Downloads files from URLs marked as not downloaded in the database",
	Long: `Iterates through URLs in the database that are not yet downloaded (downloaded = false),
ordered by date_added. It attempts to download each file to its specified
destination_path using wget with resume support (-c).

Each download attempt is logged in the 'download_runs' table. If a download
fails, it will be retried up to the number of times specified by --retries.
Upon successful download, the 'downloaded' status in the 'downloads' table
is set to true.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		fmt.Printf("Starting download process. Max retries per file: %d\n", downloadRetries)

		// Check if wget is available
		if _, err := exec.LookPath("wget"); err != nil {
			return fmt.Errorf("wget command not found in PATH. Please install wget to use the download command: %w", err)
		}

		rows, err := db.Query("SELECT id, url, destination_path FROM "+downloadsTableName+" WHERE downloaded = ? ORDER BY date_added ASC", false)
		if err != nil {
			return fmt.Errorf("failed to query pending downloads: %w", err)
		}
		defer rows.Close()

		var downloadsToProcess []struct {
			id              int64
			url             string
			destinationPath string
		}

		for rows.Next() {
			var d struct {
				id              int64
				url             string
				destinationPath string
			}
			if err := rows.Scan(&d.id, &d.url, &d.destinationPath); err != nil {
				log.Printf("Error scanning download row: %v. Skipping this entry.", err)
				continue
			}
			downloadsToProcess = append(downloadsToProcess, d)
		}
		if err := rows.Err(); err != nil {
			return fmt.Errorf("error iterating download rows: %w", err)
		}

		if len(downloadsToProcess) == 0 {
			fmt.Println("No files pending download.")
			return nil
		}

		fmt.Printf("Found %d file(s) to download.\n", len(downloadsToProcess))

		for _, item := range downloadsToProcess {
			fmt.Printf("\nProcessing Download ID %d: %s -> %s\n", item.id, item.url, item.destinationPath)

			// Create destination directory if it doesn't exist
			destDir := filepath.Dir(item.destinationPath)
			if err := os.MkdirAll(destDir, os.ModePerm); err != nil {
				log.Printf("  Failed to create destination directory '%s' for ID %d: %v. Skipping.", destDir, item.id, err)
				// Log this attempt as a failure in download_runs?
				// For now, we skip if dir creation fails, as wget would also fail.
				// A more robust solution might log this specific type of failure.
				continue
			}

			var success bool
			for attempt := 1; attempt <= downloadRetries; attempt++ {
				fmt.Printf("  Attempt %d of %d...\n", attempt, downloadRetries)
				startTime := time.Now()

				wgetCmd := exec.Command("wget", "-c", "-O", item.destinationPath, item.url)
				var stderr bytes.Buffer
				wgetCmd.Stderr = &stderr // Capture stderr for logging

				err := wgetCmd.Run() // Run the command

				endTime := time.Now()
				runSuccess := err == nil
				details := ""
				if !runSuccess {
					details = strings.TrimSpace(stderr.String())
					if err != nil { // Append exec error if different from stderr message
						if details != "" {
							details += "\n"
						}
						details += "Execution error: " + err.Error()
					}
				}

				// Log the attempt to download_runs
				_, logErr := db.Exec(
					"INSERT INTO "+downloadRunsTable+" (download_id, attempt_number, start_time, end_time, success, details) VALUES (?, ?, ?, ?, ?, ?)",
					item.id, attempt, startTime.Format(time.RFC3339), endTime.Format(time.RFC3339), runSuccess, details,
				)
				if logErr != nil {
					log.Printf("  ERROR: Failed to log download attempt for ID %d, attempt %d: %v", item.id, attempt, logErr)
					// Continue with download logic despite logging failure for this attempt
				}

				if runSuccess {
					fmt.Printf("  Download successful for ID %d.\n", item.id)
					// Update downloads table
					_, updateErr := db.Exec("UPDATE "+downloadsTableName+" SET downloaded = ? WHERE id = ?", true, item.id)
					if updateErr != nil {
						log.Printf("  ERROR: Failed to mark download ID %d as downloaded: %v", item.id, updateErr)
						// The file is downloaded, but DB update failed. This is a problematic state.
						// For now, we log and consider the download operation for this item as concluded.
					}
					success = true
					break // Exit retry loop
				} else {
					fmt.Printf("  Download failed for ID %d (Attempt %d/%d): %s\n", item.id, attempt, downloadRetries, details)
					if attempt < downloadRetries {
						fmt.Println("    Retrying...")
						time.Sleep(2 * time.Second) // Optional: wait before retrying
					}
				}
			} // End retry loop

			if !success {
				fmt.Printf("  All %d retries failed for download ID %d: %s\n", downloadRetries, item.id, item.url)
			}
		} // End loop through downloadsToProcess

		fmt.Println("\nDownload process finished.")
		return nil
	},
}

// init initializes the cobra command structure
func init() {
	rootCmd.AddCommand(loadCmd)
	rootCmd.AddCommand(listCmd)
	rootCmd.AddCommand(downloadCmd) // Add the new download command

	// Flags for loadCmd
	loadCmd.Flags().StringVarP(&inputFile, "file", "f", "", "Path to a text file containing URLs (one per line)")
	loadCmd.Flags().StringVarP(&outputDir, "output-dir", "o", "", "Directory to store downloaded files (required for load)")
	loadCmd.Flags().BoolVar(&force, "force", false, "Overwrite existing URL entry, updating date_added, destination_path, and resetting downloaded status")
	if err := loadCmd.MarkFlagRequired("output-dir"); err != nil { // output-dir is for 'load'
		log.Fatalf("Failed to mark 'output-dir' flag as required for load command: %v", err)
	}

	// Flags for listCmd
	listCmd.Flags().BoolVar(&listDownloaded, "downloaded", false, "List URLs that have been downloaded")

	// Flags for downloadCmd
	downloadCmd.Flags().IntVar(&downloadRetries, "retries", 3, "Number of times to retry a failed download")

}

// initDB initializes the SQLite database connection and creates tables if they don't exist.
func initDB(dbPath string) (*sql.DB, error) {
	d, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_busy_timeout=5000") // Added busy_timeout
	if err != nil {
		return nil, fmt.Errorf("could not open database %s: %w", dbPath, err)
	}
	if err = d.Ping(); err != nil {
		d.Close()
		return nil, fmt.Errorf("could not connect to database %s: %w", dbPath, err)
	}

	// Enable Foreign Key support if not enabled by default (good practice for SQLite)
	_, err = d.Exec("PRAGMA foreign_keys = ON;")
	if err != nil {
		d.Close()
		return nil, fmt.Errorf("could not enable foreign key support: %w", err)
	}

	createDownloadsTableSQL := `
    CREATE TABLE IF NOT EXISTS ` + downloadsTableName + ` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "url" TEXT UNIQUE NOT NULL,
        "downloaded" BOOLEAN NOT NULL DEFAULT 0,
        "date_added" TEXT NOT NULL,
        "destination_path" TEXT NOT NULL
    );`
	_, err = d.Exec(createDownloadsTableSQL)
	if err != nil {
		d.Close()
		return nil, fmt.Errorf("could not create table '%s': %w", downloadsTableName, err)
	}

	// Create the new download_runs table
	createDownloadRunsTableSQL := `
    CREATE TABLE IF NOT EXISTS ` + downloadRunsTable + ` (
        "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "download_id" INTEGER NOT NULL,
        "attempt_number" INTEGER NOT NULL,
        "start_time" TEXT NOT NULL,
        "end_time" TEXT NOT NULL,
        "success" BOOLEAN NOT NULL,
        "details" TEXT,
        FOREIGN KEY("download_id") REFERENCES "` + downloadsTableName + `"("id") ON DELETE CASCADE
    );` // Added ON DELETE CASCADE for FK
	_, err = d.Exec(createDownloadRunsTableSQL)
	if err != nil {
		d.Close()
		return nil, fmt.Errorf("could not create table '%s': %w", downloadRunsTable, err)
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
	pathParts := strings.Split(strings.Trim(u.Path, "/"), "/")

	// Handle URLs like:
	// 1. /<repo_id>/resolve/<branch>/<filename> (e.g., /TheBloke/Llama-2-7B-AWQ/resolve/main/model.safetensors)
	// 2. /<repo_id>/blob/<branch>/<filename> (less common for direct download links, but similar structure)
	// 3. /<author>/<model_name>/resolve/<branch>/<filename> (if repo_id contains '/')
	// 4. /<author>/<model_name>/raw/<filename> (older style, less common now)
	// We generally expect at least <author>/<model_name>/.../<filename> or <repo_id>/.../<filename>
	// The filename is usually the last part. The "author" can be the first part or first two parts if repo_id is like "Org/Model".

	if len(pathParts) >= 2 {
		// Simplistic approach: last part is filename, first part is "author" (or start of repo_id)
		// For repo_id like "TheBloke/Llama-2-7B-AWQ", pathParts[0] is "TheBloke", pathParts[1] is "Llama-2-7B-AWQ"
		// If the URL is /TheBloke/Llama-2-7B-AWQ/resolve/main/model.safetensors
		// pathParts = ["TheBloke", "Llama-2-7B-AWQ", "resolve", "main", "model.safetensors"]
		// We want author to be "TheBloke/Llama-2-7B-AWQ" and filename "model.safetensors"
		// Or, if the user provides outputDir, maybe just "author" as the first part of the repo_id is fine for directory structure.
		// The current code takes pathParts[0] as author.
		// Let's refine this to better match typical HF structures for directory creation.
		// If the path has "resolve" or "blob", the actual model path is usually before that.
		// Example: /tiiuae/falcon-7b/resolve/main/pytorch_model.bin
		// Author: tiiuae, Model: falcon-7b, Filename: pytorch_model.bin
		// Stored as: outputDir/tiiuae/falcon-7b/pytorch_model.bin (if outputDir/author/model_name/filename)
		// Or: outputDir/tiiuae--falcon-7b/pytorch_model.bin (if outputDir/repo_id_flat/filename)

		// For simplicity and consistency with many downloaders, let's assume the first part of the path is the "author"
		// or the organization, and the filename is the last part.
		// If the repo ID has a slash (e.g. "Org/ModelName"), pathParts[0] will be "Org".
		// The `destinationPath` in `loadCmd` is `filepath.Join(outputDir, author, filename)`.
		// This means `author` should be the primary directory name.

		author = pathParts[0]
		filename = pathParts[len(pathParts)-1]

		// Check for common intermediate path segments like "resolve", "blob", "raw"
		// and try to get a more descriptive "model name" part if available for the author string.
		// For example, if URL is /<org>/<model>/resolve/main/<file>, author could be <org>/<model>
		if len(pathParts) > 2 { // e.g. org/model/resolve/... or user/model/resolve/...
			// If pathParts[0] is an org and pathParts[1] is a model name,
			// it might be better to use "org/model" as the "author" for directory structure.
			// However, the original code used pathParts[0]. Let's stick to that for now
			// but note that this might lead to flat structures like outputDir/TheBloke/model.safetensors
			// instead of outputDir/TheBloke/Llama-2-7B-AWQ/model.safetensors if not careful.

			// The current `loadCmd` uses `author = parts[0]` and `filename = parts[len(parts)-1]`.
			// The `destinationPath` becomes `filepath.Join(outputDir, author, filename)`.
			// This is simple. If a user wants a deeper structure, they'd need to adjust `outputDir` in `loadCmd`.
			// Example: `load --output-dir /models/TheBloke/Llama-2-7B-AWQ ...`
			// Then `author` (pathParts[0]) from the URL like `/TheBloke/Llama-2-7B-AWQ/...` is "TheBloke".
			// This results in `/models/TheBloke/Llama-2-7B-AWQ/TheBloke/model.safetensors` which is not ideal.

			// Let's adjust `parseHuggingFaceURL` to return a more structured author/repo part.
			// Typically, the repo ID is the first two parts if it's an org model (e.g., "TheBloke/Llama-2-7B-GPTQ")
			// or the first part if it's a user model (e.g., "username/modelname" where "username" is pathParts[0]).
			// The crucial part for `destination_path` is how `author` is used.
			// `destinationPath := filepath.Join(outputDir, author, filename)`
			// If URL is /TheBloke/Llama-2-7B-AWQ/resolve/main/model.safetensors
			// We want author to be "TheBloke/Llama-2-7B-AWQ" for path construction.

			repoIdParts := []string{}
			for _, p := range pathParts {
				if p == "resolve" || p == "blob" || p == "raw" || p == "tree" { // "tree" for listing files
					break
				}
				repoIdParts = append(repoIdParts, p)
			}

			if len(repoIdParts) > 0 {
				// Use the full repo ID (e.g., "TheBloke/Llama-2-7B-AWQ") as the "author" for path construction.
				// This will result in outputDir/TheBloke/Llama-2-7B-AWQ/filename if repoIdParts is ["TheBloke", "Llama-2-7B-AWQ"]
				// and `filepath.Join` handles multiple elements.
				author = filepath.Join(repoIdParts...) // This will create author/modelname as part of the path
				filename = pathParts[len(pathParts)-1]
			} else {
				// Fallback or if the structure is very simple, e.g. /username/file.txt (not typical for models)
				if len(pathParts) >= 2 {
					author = pathParts[0]
					filename = pathParts[len(pathParts)-1]
				} else if len(pathParts) == 1 && !strings.Contains(pathParts[0], "/") { // e.g. /somefile.txt at root (unlikely for HF)
					author = "_" // placeholder for author if none found
					filename = pathParts[0]
				} else {
					return "", "", fmt.Errorf("URL path '%s' too short or malformed to extract repo and filename", u.Path)
				}
			}
		} else if len(pathParts) == 2 { // e.g. /username/modelname (if this is a direct file link, filename is modelname)
			author = pathParts[0]
			filename = pathParts[1]
		} else if len(pathParts) == 1 && !strings.Contains(pathParts[0], "/") { // Unlikely for HF files, more like a repo name
			return "", "", fmt.Errorf("URL path '%s' seems to be a repository root, not a file link", u.Path)
		} else {
			return "", "", fmt.Errorf("URL path '%s' does not seem to contain enough parts for repo and filename", u.Path)
		}

		if author == "" || filename == "" {
			return "", "", fmt.Errorf("could not extract non-empty author/repo ('%s') or filename ('%s') from path '%s'", author, filename, u.Path)
		}
		// Sanitize filename to prevent path traversal issues, though filepath.Join should handle it.
		filename = filepath.Base(filename)
		if filename == "." || filename == ".." {
			return "", "", fmt.Errorf("extracted filename ('%s') is invalid", filename)
		}

		return author, filename, nil
	}
	return "", "", fmt.Errorf("URL path '%s' is empty or invalid after splitting", u.Path)
}

// main is the entry point of the application
func main() {
	if err := rootCmd.Execute(); err != nil {
		// Cobra already prints the error, so just exit
		os.Exit(1)
	}
}
