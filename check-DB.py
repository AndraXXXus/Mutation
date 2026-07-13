import sqlite3


def check_experiment_results(db_path="experiment_results.db"):
    # Connect to the local SQLite database
    conn = sqlite3.connect(db_path)
    # Set the row factory to sqlite3.Row to easily access columns by name
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # 1. Get total number of elements (rows) in the table
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total_rows = cursor.fetchone()[0]

        # 2. Get total number of rows with status 'COMPLETED'
        cursor.execute(
            "SELECT COUNT(*) FROM experiments WHERE status = 'COMPLETED'"
        )
        completed_rows = cursor.fetchone()[0]

        # Print the counts
        print(f"Total rows in DB: {total_rows}")
        print(f"Total 'COMPLETED' rows: {completed_rows}")
        print(f" 'COMPLETED' %: {completed_rows / total_rows * 100:.2f}%")
        print("-" * 40)


    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        # Always close the database connection
        conn.close()

def check_experiment_results2(db_path="experiment_results.db"):
    # Connect to the local SQLite database
    conn = sqlite3.connect(db_path)
    # Set the row factory to sqlite3.Row to easily access columns by name
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # 1. Get total number of elements (rows) in the table
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total_rows = cursor.fetchone()[0]

        # 2. Get total number of rows with status 'COMPLETED'
        cursor.execute(
            "SELECT COUNT(*) FROM experiments WHERE status = 'PENDING'"
        )
        completed_rows = cursor.fetchone()[0]

        # Print the counts
        print(f"Total rows in DB: {total_rows}")
        print(f"Total 'PENDING' rows: {completed_rows}")
        print(f" 'PENDING' %: {completed_rows / total_rows * 100:.2f}%")
        print("-" * 40)


    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        # Always close the database connection
        conn.close()


if __name__ == "__main__":
    check_experiment_results()
    check_experiment_results2()