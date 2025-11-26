"""
Action space definitions and execution engine for the ToM Agent workflow controller.

This module defines the available actions, response models, and execution engine
that the agent can use to interact with files, process sessions, and manage user models.
"""

import asyncio
import json
import logging
from typing import Any, Optional, List
from datetime import datetime

# Try to import BM25 dependencies - fall back to pure Python if unavailable
# The pure Python implementation (bm25_fallback.py) provides:
# - Similar API to bm25s for seamless integration
# - Limited to 10 most recent conversations for performance
# - Simplified Porter stemming approximation (~80% accuracy)
# - Standard BM25 ranking algorithm
# Install with: pip install tom-swe[search] for full bm25s support
try:
    import bm25s
    import Stemmer

    HAS_BM25S = True
except ImportError:
    HAS_BM25S = False
    from tom_swe.generation.bm25_fallback import (
        BM25 as bm25s_BM25,
        SimpleStemmer,
        tokenize_corpus,
    )

    # Create mock bm25s module for consistent API
    class _Bm25sFallback:
        BM25 = bm25s_BM25

        @staticmethod
        def tokenize(text: Any, **kwargs: Any) -> Any:
            if isinstance(text, str):
                return tokenize_corpus([text], **kwargs)[0]
            else:
                return tokenize_corpus(text, **kwargs)

    bm25s = _Bm25sFallback()

from tom_swe.generation.dataclass import (
    ActionType,
    ReadFileParams,
    SearchFileParams,
    UpdateJsonFieldParams,
    AnalyzeSessionParams,
    InitializeUserProfileParams,
    GenerateSuggestionsParams,
    GenerateSleepSummaryParams,
    RagSearchParams,
    SessionAnalysis,
)
from tom_swe.memory.locations import (
    get_overall_user_model_filename,
    get_cleaned_sessions_dir,
    get_cleaned_session_filename,
    get_session_model_filename,
    get_session_models_dir,
)
from tom_swe.memory.store import FileStore

try:
    from tom_swe.logging_config import get_tom_swe_logger

    logger = get_tom_swe_logger(__name__)
except ImportError:
    # Fallback for standalone use
    logger = logging.getLogger(__name__)


class ActionExecutor:
    """Executes actions for the ToM Agent workflow controller."""

    def __init__(
        self,
        user_id: str,
        file_store: FileStore,
        agent_context: Optional[Any] = None,
    ):
        """
        Initialize the action executor.

        Args:
            agent_context: Reference to the ToM agent for accessing its methods and state
            file_store: FileStore for I/O operations
        """
        self.agent_context = agent_context
        self.file_store = file_store
        self.user_id = user_id

    def execute_action(
        self, action: ActionType, parameters: Any
    ) -> str | GenerateSuggestionsParams | GenerateSleepSummaryParams:
        """
        Execute a specific action with given parameters.

        Args:
            action: The action to execute
            parameters: Parameters for the action

        Returns:
            Result of the action execution
        """
        logger.info(f"🎯 Executing action: {action.value}")
        logger.info(f"📋 Parameters: {parameters}")

        # Handle final response actions - these contain the response data in parameters
        if action == ActionType.GENERATE_SUGGESTIONS:
            logger.info("📤 Final response action: returning suggestions data")
            assert isinstance(parameters, GenerateSuggestionsParams)
            return parameters  # Return the structured response data directly
        elif action == ActionType.GENERATE_SLEEP_SUMMARY:
            logger.info("📤 Final response action: returning sleep summary data")
            assert isinstance(parameters, GenerateSleepSummaryParams)
            return parameters  # Return the structured response data directly

        # Handle regular execution actions
        if action == ActionType.READ_FILE:
            return self._action_read_file(parameters)
        elif action == ActionType.SEARCH_FILE:
            return self._action_search_file(parameters)
        elif action == ActionType.UPDATE_JSON_FIELD:
            return self._action_update_json_field(parameters)
        elif action == ActionType.ANALYZE_SESSION:
            return self._action_analyze_session(parameters)
        elif action == ActionType.INITIALIZE_USER_PROFILE:
            return self._action_initialize_user_profile(parameters)
        elif action == ActionType.RAG_SEARCH:
            return self._action_rag_search(parameters)
        else:
            return f"Action {action.value} not implemented yet"

    # Action implementations
    def _action_read_file(self, params: ReadFileParams) -> str:
        """Read a file."""
        try:
            content = self.file_store.read(params.file_path)[
                params.character_start : params.character_end
            ]
            return content
        except Exception as e:
            return f"Error reading {params.file_path}: {str(e)}"

    def _get_content_by_scope(
        self,
        search_scope: str,
        latest_first: bool = True,
        chunk_size: int = 5000,
        limit: int = 50,
    ) -> List[tuple[str, str]]:
        """Get file content by search scope, optionally sorted by date. Returns list of (file_path, content) tuples."""
        # Limit to 10 conversations when using pure Python BM25
        if not HAS_BM25S and limit > 10:
            logger.warning(
                f"Using pure Python BM25 fallback - limiting search to 10 most recent conversations "
                f"(requested {limit}). Install with 'pip install tom-swe[search]' for full functionality."
            )
            limit = 10

        try:
            if search_scope == "cleaned_sessions":
                files = self.file_store.list(get_cleaned_sessions_dir(self.user_id))
            elif search_scope == "session_analyses":
                files = self.file_store.list(get_session_models_dir(self.user_id))
            elif search_scope == "user_profiles":
                files = [get_overall_user_model_filename(self.user_id)]
            else:
                files = []
            # Read content and prepare for sorting
            file_content_pairs = []
            for file_path in files:
                try:
                    content = self.file_store.read(file_path)
                    file_content_pairs.append((file_path, content))
                except Exception:
                    continue

            # Sort by last_updated if latest_first
            if latest_first and file_content_pairs:
                file_times = []
                for file_path, content in file_content_pairs:
                    try:
                        data = json.loads(content)
                        last_updated = data.get("last_updated", "1970-01-01")
                        file_times.append((file_path, content, last_updated))
                    except Exception:
                        file_times.append((file_path, content, "1970-01-01"))
                file_times.sort(key=lambda x: str(x[2]), reverse=True)
                file_content_pairs = [(f[0], f[1]) for f in file_times]

            if search_scope == "cleaned_sessions":
                chunked_content_pairs = []
                for file_path, content in file_content_pairs:
                    for i in range(0, len(content), chunk_size):
                        if i == 0:
                            continue  # we skip the first chunk because it's the system prompt
                        chunked_content_pairs.append(
                            (file_path, content[i : i + chunk_size])
                        )
                file_content_pairs = chunked_content_pairs
            return file_content_pairs[:limit]
        except Exception:
            return []

    def _string_search(self, params: SearchFileParams) -> str:
        """Original string-based search implementation."""
        try:
            # Use consolidated content loading (reads files once)
            file_content_pairs = self._get_content_by_scope(
                params.search_scope, params.latest_first, 50
            )

            results = []
            for file_path, content in file_content_pairs:
                if params.query.lower() in content.lower():
                    lines = [
                        line.strip()
                        for line in content.split("\n")
                        if params.query.lower() in line.lower()
                    ][:2]
                    if lines:
                        results.append(f"{file_path}:\n" + "\n".join(lines))
                        if len(results) >= params.max_results:
                            break

            return (
                f"Found {len(results)} files:\n\n" + "\n\n".join(results)
                if results
                else f"No files found containing '{params.query}'"
            )
        except Exception as e:
            return f"String search error: {str(e)}"

    def _action_search_file(self, params: SearchFileParams) -> str:
        """
        Search within files using BM25 or string matching.

        BM25 search uses bm25s library if available (install with tom-swe[search]),
        otherwise falls back to pure Python implementation with 10-conversation limit.

        Args:
            params: Search parameters including query, scope, and method

        Returns:
            Formatted search results with scores and snippets
        """
        if params.search_method == "string_match":
            return self._string_search(params)

        # BM25 search (default)
        try:
            # Log which implementation is being used
            if not HAS_BM25S:
                logger.info(
                    "Using pure Python BM25 fallback (limited to 10 conversations)"
                )

            # Use consolidated file loading with date sorting
            file_content_pairs = self._get_content_by_scope(
                params.search_scope, params.latest_first, params.chunk_size, 50
            )

            # Extract document contents
            corpus = []
            file_paths = []
            for file_path, content in file_content_pairs:
                corpus.append(content)
                file_paths.append(file_path)

            if not corpus:
                return f"No files found in scope '{params.search_scope}'"

            # Create stemmer - use appropriate type based on availability
            if HAS_BM25S:
                stemmer = Stemmer.Stemmer("english")
            else:
                stemmer = SimpleStemmer()

            corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

            # Build BM25 index
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)

            # Search - tokenize query with same parameters
            query_tokens = bm25s.tokenize(params.query, stopwords="en", stemmer=stemmer)
            results, scores = retriever.retrieve(query_tokens, k=params.max_results)

            # Format results
            formatted_results = []
            for i in range(results.shape[1]):
                if i >= len(file_paths):
                    break
                doc_idx = results[0, i]
                score = scores[0, i]
                if doc_idx >= len(file_paths):
                    continue
                file_path = file_paths[doc_idx]

                # Get relevant snippet
                content = corpus[doc_idx]
                snippet = content[:5000] + "..." if len(content) > 5000 else content

                formatted_results.append(
                    f"[Score: {score:.2f}] {file_path}:\n{snippet}"
                )

            return (
                f"Found {len(formatted_results)} files (BM25 ranked):\n\n"
                + "\n\n".join(formatted_results)
                if formatted_results
                else f"No relevant files found for '{params.query}'"
            )

        except Exception as e:
            logger.warning(f"BM25 search failed: {e}, using string search")
            return self._string_search(params)

    def _action_update_json_field(self, params: UpdateJsonFieldParams) -> str:
        """Update a specific JSON field."""
        field_path = get_overall_user_model_filename(self.user_id)
        try:
            # Read existing data or create new
            try:
                data = json.loads(self.file_store.read(field_path))
            except Exception:
                if params.create_if_missing:
                    data = {}
                else:
                    return f"File not found: {field_path}"

            # Navigate to field using dot notation
            current = data
            field_parts = params.field_path.split(".")

            for part in field_parts[:-1]:
                if part not in current:
                    if params.create_if_missing:
                        current[part] = {}
                    else:
                        return f"Field path not found: {part}"
                current = current[part]

            # Handle list operations
            final_field = field_parts[-1]
            old_value = current.get(final_field)

            if isinstance(old_value, list) and params.list_operation in [
                "append",
                "remove",
            ]:
                if params.list_operation == "append":
                    if params.new_value not in old_value:  # Avoid duplicates
                        old_value.append(params.new_value)
                elif params.list_operation == "remove":
                    if isinstance(params.new_value, int):
                        # Remove by index
                        if 0 <= params.new_value < len(old_value):
                            old_value.pop(params.new_value)
                    else:
                        # Remove by value
                        if params.new_value in old_value:
                            old_value.remove(params.new_value)
            else:
                # Default: replace the field
                current[final_field] = params.new_value

            # Write back
            data["last_updated"] = datetime.now().isoformat()
            self.file_store.write(field_path, json.dumps(data, indent=2))

            return f"Updated {field_path}: {params.list_operation} {params.new_value}"

        except Exception as e:
            return f"Update error: {str(e)}"

    def _action_analyze_session(self, params: AnalyzeSessionParams) -> str:
        """Process a batch of sessions using ToM analyzer."""
        user_id = params.user_id
        session_batch = params.session_batch

        if not session_batch:
            return "Error: session_batch parameter is required"

        logger.info(
            f"🧠 Processing batch of {len(session_batch)} sessions for user {user_id}"
        )

        # Get the ToM analyzer from agent context
        if not self.agent_context or not hasattr(self.agent_context, "tom_analyzer"):
            return "Error: ToM analyzer not available in agent context"

        tom_analyzer = self.agent_context.tom_analyzer

        # Load all session data first
        session_data_list = []
        for session_id in session_batch:
            try:
                session_file = get_cleaned_session_filename(
                    session_id, user_id if user_id else None
                )
                content = self.file_store.read(session_file)
                session_data = json.loads(content)
                session_data_list.append(session_data)
            except Exception as e:
                logger.error(f"Error loading session {session_id}: {e}")
                continue

        async def _analyze() -> Any:
            return await tom_analyzer.process_session_batch(session_data_list)

        session_summaries = asyncio.run(_analyze())
        # Save session analyses and prepare result
        session_dumps = []
        for session_analysis in session_summaries:
            session_dump = session_analysis.model_dump()
            session_dumps.append(session_dump)
            session_file = get_session_model_filename(
                session_analysis.session_id, user_id
            )
            self.file_store.write(session_file, json.dumps(session_dump, indent=2))

        return json.dumps(session_dumps, indent=2)

    def _action_initialize_user_profile(
        self, params: InitializeUserProfileParams
    ) -> str:
        """Initialize and save user analysis using tom_module."""
        user_id = params.user_id
        if not self.agent_context or not hasattr(self.agent_context, "tom_analyzer"):
            return "Error: ToM analyzer not available in agent context"
        tom_analyzer = self.agent_context.tom_analyzer

        # Load all session analyses from files
        session_analyses = []

        for filename in self.file_store.list(get_session_models_dir(user_id)):
            if filename.endswith(".json"):
                content = self.file_store.read(filename)
                session_data = json.loads(content)
                # Convert back to SessionAnalysis object
                session_analysis = SessionAnalysis(**session_data)
                session_analyses.append(session_analysis)

        # Call ToM analyzer to initialize user analysis
        async def _initialize() -> Any:
            return await tom_analyzer.initialize_user_analysis(session_analyses)

        user_analysis = asyncio.run(_initialize())

        # Save user analysis to file and prepare result
        user_analysis_dump = user_analysis.model_dump()
        user_model_file = get_overall_user_model_filename(user_id)
        self.file_store.write(user_model_file, json.dumps(user_analysis_dump, indent=2))

        return json.dumps(user_analysis_dump, indent=2)

    def _action_rag_search(self, params: RagSearchParams) -> str:
        """Search for relevant context using RAG."""
        # TODO: Implement RAG functionality
        return f"RAG search for '{params.query}' (k={params.k}) - not implemented yet"
