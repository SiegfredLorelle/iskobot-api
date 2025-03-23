from app.config import Config
import psycopg2

def get_db_connection() -> psycopg2.extensions.connection:
    """Create a direct connection to Supabase PostgreSQL database."""
    return psycopg2.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        user=Config.DB_USER,
        password=Config.DB_PASS,
        database=Config.DB_NAME,
    )


# Test connection
if __name__ == "__main__":
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Test simple query
        cursor.execute("SELECT 1;")
        
        # Test Postgres version
        print("✅ Connection successful!")
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()
        print(f"PostgreSQL version: {pg_version[0]}")
        conn.close()
        
    except Exception as e:
        print("❌ Connection failed!")
        print(f"Error: {str(e)}")