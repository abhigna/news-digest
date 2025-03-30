# News Digest Project Guidelines

## Commands
- Run collection: `python src/cli.py --collect`
- Filter articles: `python src/cli.py --filter [--days N] [--no-cache]`
- Summarize: `python src/cli.py --summarize`
- Compile digest: `python src/cli.py --compile [--days N]`
- Send email: `python src/cli.py --email "user@example.com"`
- Run evaluation app: `streamlit run src/judge_app.py`

## Code Style
- Python 3.8+ with type hints (use `typing` module)
- Follow PEP 8 naming: snake_case for variables/functions, CamelCase for classes
- Use Pydantic for data models and validation
- Imports: standard library first, third-party second, local modules last
- Triple-quote docstrings for all classes and functions
- Comprehensive exception handling with try/except blocks
- Use logging (not print) for debug/info output
- Keep single responsibility principle in mind for module organization

## Error Handling
- Catch specific exceptions, not broad Exception class where possible
- Include traceback in critical failure logs
- Use centralized error logging with proper log levels
- Handle API failures gracefully with appropriate user feedback