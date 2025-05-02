import requests
import logging
import time # Import time for sleep
from notion_second_brain import config

logger = logging.getLogger(__name__)

class NotionClient:
    """Client to interact with the Notion API."""

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28" # Specify the Notion API version
    MAX_RETRIES = 3 # Maximum number of retries for rate limits
    DEFAULT_RETRY_AFTER = 1 # Default seconds to wait if Retry-After header is missing

    def __init__(self, token: str = config.NOTION_TOKEN):
        if not token:
            raise ValueError("Notion API token is missing.")
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Notion-Version": self.NOTION_VERSION,
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _request(self, method: str, endpoint: str, **kwargs):
        """Helper method for making requests to the Notion API with retry logic for rate limits."""
        url = f"{self.BASE_URL}/{endpoint}"
        retries = 0
        while retries <= self.MAX_RETRIES:
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                return response.json()
            
            except requests.exceptions.HTTPError as e:
                # Check specifically for rate limit error (429)
                if e.response.status_code == 429 and retries < self.MAX_RETRIES:
                    retry_after = e.response.headers.get('Retry-After')
                    try:
                        wait_time = int(retry_after) if retry_after else self.DEFAULT_RETRY_AFTER
                    except (ValueError, TypeError):
                        wait_time = self.DEFAULT_RETRY_AFTER
                    
                    retries += 1
                    logger.warning(
                        f"Rate limit exceeded (429). Retrying after {wait_time} seconds... (Attempt {retries}/{self.MAX_RETRIES}) for {method} {endpoint}"
                    )
                    time.sleep(wait_time)
                    continue # Retry the loop
                else:
                    # For other HTTP errors or if max retries exceeded for 429
                    logger.error(f"Notion API HTTP error: {e}")
                    logger.error(f"Response status code: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text}")
                    raise # Re-raise the original HTTPError
            
            except requests.exceptions.RequestException as e:
                # For non-HTTP request errors (connection, timeout, etc.)
                logger.error(f"Notion API request failed: {e}")
                if e.response is not None: # Might not have a response object
                    logger.error(f"Response status code: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text}")
                raise # Re-raise the original RequestException
            
            except Exception as e:
                # Catch any other unexpected errors during the request/response handling
                logger.error(f"An unexpected error occurred during API request: {e}")
                raise # Re-raise the unexpected exception

        # Should not be reached if MAX_RETRIES > 0, but as a safeguard
        logger.error(f"Max retries exceeded for {method} {endpoint}")
        raise requests.exceptions.RequestException(f"Max retries exceeded for {method} {endpoint}")

    def test_connection(self, database_id: str = config.DATABASE_ID):
        """Tests the connection to the Notion API by retrieving database info."""
        if not database_id:
            raise ValueError("Database ID is missing. Cannot test connection.")
        
        endpoint = f"databases/{database_id}"
        logger.info(f"Testing Notion connection by retrieving database: {database_id}")
        try:
            response_data = self._request("GET", endpoint)
            logger.info(f"Successfully connected to Notion and retrieved database '{response_data.get('title', [{}])[0].get('plain_text', 'Untitled')}'")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Notion or retrieve database {database_id}. Error: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during connection test: {e}")
            return False

    def retrieve_database_schema(self, database_id: str = config.DATABASE_ID):
        """Retrieves the schema (properties) of a Notion database.

        Args:
            database_id: The ID of the Notion database.

        Returns:
            A dictionary representing the database object, including its properties.
            Returns None if the database cannot be retrieved.
        """
        if not database_id:
            raise ValueError("Database ID is missing. Cannot retrieve schema.")

        endpoint = f"databases/{database_id}"
        logger.info(f"Retrieving schema for Notion database: {database_id}")
        try:
            schema_data = self._request("GET", endpoint)
            logger.info(f"Successfully retrieved schema for database '{schema_data.get('title', [{}])[0].get('plain_text', 'Untitled')}'")
            return schema_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve schema for database {database_id}. Error: {e}")
            return None # Or re-raise depending on desired error handling
        except Exception as e:
            logger.error(f"An unexpected error occurred during schema retrieval for database {database_id}: {e}")
            return None # Or re-raise

    def query_database(self, database_id: str = config.DATABASE_ID, filter_params: dict | None = None, sort_params: list | None = None):
        """Queries a Notion database and retrieves all entries, handling pagination.

        Args:
            database_id: The ID of the Notion database to query.
            filter_params: Optional Notion API filter object (https://developers.notion.com/reference/post-database-query-filter).
            sort_params: Optional Notion API sort object (https://developers.notion.com/reference/post-database-query-sort).

        Returns:
            A list of all page objects retrieved from the database.
        """
        if not database_id:
            raise ValueError("Database ID is missing. Cannot query database.")
        
        endpoint = f"databases/{database_id}/query"
        all_results = []
        has_more = True
        next_cursor = None

        logger.info(f"Querying Notion database: {database_id}")

        while has_more:
            payload = {}
            if filter_params:
                payload['filter'] = filter_params
            if sort_params:
                payload['sorts'] = sort_params
            if next_cursor:
                payload['start_cursor'] = next_cursor

            try:
                # Use json parameter for POST request body
                response_data = self._request("POST", endpoint, json=payload) 
                all_results.extend(response_data.get("results", []))
                has_more = response_data.get("has_more", False)
                next_cursor = response_data.get("next_cursor")
                if has_more:
                    logger.debug(f"Fetching next page for database {database_id}...")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to query database {database_id}. Error: {e}")
                # Depending on requirements, you might want to return partial results or raise the exception
                break # Stop pagination on error
            except Exception as e:
                logger.error(f"An unexpected error occurred during database query: {e}")
                break # Stop pagination on error

        logger.info(f"Retrieved {len(all_results)} entries from database {database_id}.")
        return all_results

    def retrieve_block_children(self, block_id: str):
        """Retrieves all block children for a given block ID (e.g., a page ID), handling pagination.

        Args:
            block_id: The ID of the block (page, database, etc.) whose children to retrieve.

        Returns:
            A list of all block objects that are children of the given block ID.
        """
        endpoint = f"blocks/{block_id}/children"
        all_blocks = []
        has_more = True
        next_cursor = None

        logger.debug(f"Retrieving block children for block: {block_id}")

        while has_more:
            params = {}
            if next_cursor:
                params['start_cursor'] = next_cursor
            
            try:
                # GET request with query parameters
                response_data = self._request("GET", endpoint, params=params)
                all_blocks.extend(response_data.get("results", []))
                has_more = response_data.get("has_more", False)
                next_cursor = response_data.get("next_cursor")
                if has_more:
                    logger.debug(f"Fetching next page of blocks for {block_id}...")

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to retrieve block children for {block_id}. Error: {e}")
                break # Stop pagination on error
            except Exception as e:
                logger.error(f"An unexpected error occurred retrieving block children for {block_id}: {e}")
                break # Stop pagination on error
        
        logger.debug(f"Retrieved {len(all_blocks)} blocks for block {block_id}.")
        return all_blocks

# Example usage (optional, can be removed or put under if __name__ == '__main__'):
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     try:
#         client = NotionClient()
#         if client.test_connection():
#             print("Notion connection successful!")
#         else:
#             print("Notion connection failed.")
#     except ValueError as e:
#         print(e)
#     except Exception as e:
#         print(f"An error occurred: {e}") 