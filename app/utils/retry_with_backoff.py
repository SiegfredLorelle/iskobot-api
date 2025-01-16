import asyncio
from google.api_core.exceptions import ResourceExhausted

async def retry_with_backoff(func, retries=5, backoff_in_seconds=1):
    """Retries a coroutine with exponential backoff."""
    for attempt in range(retries):
        try:
            return await func()
        except ResourceExhausted as e:
            if attempt < retries - 1:
                wait_time = backoff_in_seconds * (2 ** attempt)  # Exponential backoff
                print(f"Quota exceeded. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print("Max retries reached.")
                raise e
