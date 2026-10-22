# Test-only: links http_request into the test binaries (not the released llm extension).
# Used via: make release EXTRA_EXTENSION_CONFIGS=test/configs/http_request.cmake
duckdb_extension_load(http_request
    GIT_URL https://github.com/midwork-finds-jobs/duckdb_http_request
    GIT_TAG cffc7f70a82b825dd21cb9f10b375ac004f81f45
)
duckdb_extension_statically_link(http_request)
