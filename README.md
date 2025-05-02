# notion-second-brain

A project to extract journal entries from Notion, process them into a queryable format, and build an interface to interact with your personal knowledge.

## Project Overview

This project aims to build a system that can:
1. Extract journal entries from Notion via their API
2. Organize and process the data by various time periods (day, week, month, year, all time)
3. Store the data in a format suitable for querying by LLMs
4. (Future) Create a chat interface to interact with your journal data

## Technology Stack

- **Backend**: Python (Flask or FastAPI for future API endpoints)
- **Data Processing**: Python (requests, json, datetime)
- **Storage**: JSON files initially, potential database integration later
- **LLM Integration**: OpenAI API or Anthropic's Claude API
- **Frontend** (future): React or Next.js

## Development Phases

### Phase 1: Notion Data Extraction (MVP)

#### 1.1 Setup Project Structure
- Initialize Git repository
- Create virtual environment
- Setup basic project scaffolding
- Create README.md with project overview

#### 1.2 Notion API Integration
- Set up Notion integration
- Connect to Notion API
- Retrieve database ID and test connection
- Implement basic query to fetch journal entries

#### 1.3 Data Extraction and Processing
- Extract text content from journal entries
- Parse and clean the data
- Implement date-based filtering (day, week, month, year)
- Create data structures to organize content

#### 1.4 JSON Export
- Create functionality to export data as JSON
- Organize JSON by time periods
- Implement file naming conventions
- Add metadata to exported files

### Phase 2: Data Processing and Storage

#### 2.1 Enhanced Data Processing
- Implement better text extraction for various block types
- Add support for handling images references
- Create summary statistics for journal entries
- Implement topic extraction or categorization

#### 2.2 Storage System
- Design a more robust storage system
- Implement incremental sync functionality
- Add versioning for exported data
- Create a data update pipeline

#### 2.3 Data Transformation for LLMs
- Create embedding generation for entries
- Implement chunking strategies
- Design metadata structure for improved retrieval
- Build utility functions for data preparation

### Phase 3: Query Interface

#### 3.1 Command Line Interface
- Build a CLI to trigger extraction and sync
- Add query capabilities from command line
- Implement basic search functionality
- Create reporting and statistics features

#### 3.2 LLM Integration
- Integrate with OpenAI or Claude API
- Implement RAG (Retrieval Augmented Generation) pattern
- Create prompt templates for different query types
- Build context window management

### Phase 4: Web Interface (Future)

#### 4.1 Backend API
- Design RESTful API
- Implement authentication
- Create endpoints for data retrieval
- Build query processing middleware

#### 4.2 Frontend Development
- Set up React/Next.js application
- Create UI for chat interface
- Implement data visualization components
- Build user authentication

## Implementation Plan

### Week 1: Project Setup & Basic Notion Integration
- [ ] Initialize project repository
- [ ] Create virtual environment and install dependencies
- [ ] Setup Notion integration
- [ ] Implement basic API connection
- [ ] Test connection with a simple query
- [ ] Write comprehensive README.md

### Week 2: Core Data Extraction
- [ ] Build database query functionality
- [ ] Implement pagination for large databases
- [ ] Create text extraction for different block types
- [ ] Build date parsing and filtering
- [ ] Implement basic error handling

### Week 3: Data Organization & Export
- [ ] Develop time-based filtering (day, week, month, year)
- [ ] Build JSON export functionality
- [ ] Implement file organization structure
- [ ] Create metadata generation
- [ ] Add command line arguments for extraction options

### Week 4: Testing & Refinement
- [ ] Write unit tests for API interaction
- [ ] Create integration tests for data extraction
- [ ] Implement edge case handling
- [ ] Add logging and monitoring
- [ ] Document code and API usage
- [ ] Release MVP version

## Cursor Development Guidelines

### Project Organization

```
notion-second-brain/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── docs/
│   └── api_usage.md
├── tests/
│   ├── __init__.py
│   ├── test_notion_api.py
│   └── test_data_processing.py
└── notion_second_brain/
    ├── __init__.py
    ├── config.py
    ├── notion/
    │   ├── __init__.py
    │   ├── api.py
    │   └── extractors.py
    ├── processing/
    │   ├── __init__.py
    │   ├── filters.py
    │   └── transformers.py
    ├── storage/
    │   ├── __init__.py
    │   └── json_storage.py
    └── cli.py
```

### Development Rules

1. **Incremental Development**: Follow the task breakdown and work on one component at a time
2. **Test-Driven Development**: Write tests before implementing features
3. **Documentation**: Document code and keep README updated
4. **Environment Variables**: Store all sensitive information in environment variables
5. **Error Handling**: Implement robust error handling for API interactions
6. **Git Workflow**: Commit frequently with descriptive messages
7. **Task Management**: Create GitHub issues for each task and track progress

## Getting Started

These steps assume you have Python 3 and `git` installed.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url> # Replace with your repo URL if pushed to GitHub/etc.
    cd notion-second-brain
    ```
    *(If you didn't clone, ensure you are in the `notion-second-brain` directory)*

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create the environment (only needs to be done once)
    python3 -m venv venv 

    # Activate the environment (needs to be done every time you open a new terminal)
    # On macOS/Linux (bash/zsh):
    source venv/bin/activate
    # On Windows (Cmd):
    # venv\Scripts\activate.bat
    # On Windows (PowerShell):
    # venv\Scripts\Activate.ps1
    ```
    *Your terminal prompt should now indicate the active environment, e.g., `(venv)`.*

3.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    # or just 'pip' if pip3 is not found within the activated venv
    # pip install -r requirements.txt 
    ```

4.  **Set up Notion Integration:**
    - Go to [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations) and create a new **internal** integration.
    - Copy the "Internal Integration Token".
    - Find the ID of the Notion database you want to use.
      *(Hint: The database ID is the part of the URL between your workspace name and the `?v=`..., e.g., `https://www.notion.so/yourworkspace/DATABASEID?v=...`)*
    - Share the database with the integration you created (click the `...` menu on the database page > "Add connections" > find your integration).

5.  **Configure Environment Variables:**
    - Make a copy of the `.env.example` file and name it `.env`:
      ```bash
      cp .env.example .env
      ```
    - Open the `.env` file and paste your Notion token and database ID:
      ```dotenv
      NOTION_TOKEN=secret_YOUR_NOTION_TOKEN
      NOTION_DATABASE_ID=YOUR_DATABASE_ID
      ```

6.  **Test the Setup:**
    ```bash
    python cli.py --test-connection
    ```
    *You should see a "Notion connection successful!" message.*

7.  **Run an Extraction:**
    ```bash
    # Example: Extract all entries
    python cli.py

    # Example: Extract entries from today
    python cli.py -p day

    # Example: Extract entries from a specific week (using verbose logging)
    python cli.py -p week --date 2023-10-23 -v
    ```
    *Output files will be saved in the `output/` directory by default.*

## Example Implementation Plan for Cursor

When working in Cursor, you can use these prompts to guide the implementation:

1. "Create the basic project structure for notion-second-brain"
2. "Implement the Notion API connection module"
3. "Create the journal entry extraction functionality"
4. "Implement date-based filtering for journal entries"
5. "Build the JSON export module with time-period organization"
6. "Create a CLI interface for the extraction process"
7. "Add error handling and logging to the Notion API module"
8. "Implement unit tests for the data extraction process"

Remember to:
- Use Cursor's AI features to help with implementation details
- Create tasks for each component before implementing
- Run tests frequently to verify functionality
- Document your code and update the README as you progress