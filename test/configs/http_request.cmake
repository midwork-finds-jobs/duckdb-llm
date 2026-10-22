# Test-only: links http_request into the test binaries (not the released llm extension).
# Used via: make release EXTRA_EXTENSION_CONFIGS=test/configs/http_request.cmake
duckdb_extension_load(http_request
    GIT_URL https://github.com/midwork-finds-jobs/duckdb_http_request
    GIT_TAG 40d1d3219ce7ad7bcc9a027d0d6ea0b091590dfa
)
