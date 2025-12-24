#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "providers.hpp"
#include <string>
#include <vector>

namespace duckdb {

// Batch job status (normalized across providers)
enum class BatchStatus { PENDING, VALIDATING, IN_PROGRESS, FINALIZING, COMPLETED, FAILED, EXPIRED, CANCELLED, UNKNOWN };

// Convert status to string
std::string BatchStatusToString(BatchStatus status);

// Parse status from provider strings
BatchStatus ParseOpenAIBatchStatus(const std::string &status);
BatchStatus ParseGeminiBatchStatus(const std::string &state);

// Batch job info structure
struct BatchJobInfo {
	std::string batch_id;
	std::string provider;
	std::string batch_name;
	std::string model;
	BatchStatus status;
	std::string input_file_id;
	std::string output_file_id;
	std::string error_file_id;
	int total_requests = 0;
	int completed_requests = 0;
	int failed_requests = 0;
	timestamp_t created_at;
	timestamp_t updated_at;
	timestamp_t expires_at;
};

// Batch result entry
struct BatchResultEntry {
	std::string request_id;
	std::string response;
	std::string status; // "ok" or "error"
	std::string error_message;
};

// ============================================================================
// Persistence Helpers
// ============================================================================

// Ensure batch tables exist in the database
void EnsureBatchTablesExist(ClientContext &context);

// Ensure batch cache table exists
void EnsureBatchCacheTableExists(ClientContext &context);

// Save a new batch job to the jobs table
void SaveBatchJob(ClientContext &context, const BatchJobInfo &job);

// Update an existing batch job
void UpdateBatchJob(ClientContext &context, const BatchJobInfo &job);

// Get batch job by ID
bool GetBatchJob(ClientContext &context, const std::string &batch_id, BatchJobInfo &job);

// Save request ID mappings
void SaveBatchRequests(ClientContext &context, const std::string &batch_id,
                       const std::vector<std::pair<std::string, std::string>> &id_mappings);

// Cache batch results
void CacheBatchResults(ClientContext &context, const std::string &batch_id,
                       const std::vector<BatchResultEntry> &results);

// Get cached results
std::vector<BatchResultEntry> GetCachedResults(ClientContext &context, const std::string &batch_id);

// Check if results are cached
bool HasCachedResults(ClientContext &context, const std::string &batch_id);

// ============================================================================
// JSONL Builders
// ============================================================================

// Build OpenAI batch JSONL content
// requests: vector of (request_id, prompt) pairs
std::string BuildOpenAIBatchJsonl(const std::vector<std::pair<std::string, std::string>> &requests,
                                  const std::string &model, double temperature, int max_tokens,
                                  const std::string &system_prompt);

// Build Gemini batch JSONL content
std::string BuildGeminiBatchJsonl(const std::vector<std::pair<std::string, std::string>> &requests,
                                  const std::string &model, double temperature);

// ============================================================================
// OpenAI Batch Operations
// ============================================================================

// Upload JSONL file to OpenAI Files API
// Returns file_id on success, throws on error
std::string OpenAIUploadBatchFile(ClientContext &context, const std::string &jsonl_content, const std::string &api_key,
                                  const std::string &base_url);

// Create a batch job
// Returns batch_id on success, throws on error
std::string OpenAICreateBatch(ClientContext &context, const std::string &file_id, const std::string &api_key,
                              const std::string &base_url);

// Get batch status from API
BatchJobInfo OpenAIGetBatchStatus(ClientContext &context, const std::string &batch_id, const std::string &api_key,
                                  const std::string &base_url);

// Download results file content
std::string OpenAIDownloadFile(ClientContext &context, const std::string &file_id, const std::string &api_key,
                               const std::string &base_url);

// Cancel a batch
bool OpenAICancelBatch(ClientContext &context, const std::string &batch_id, const std::string &api_key,
                       const std::string &base_url);

// Parse OpenAI results JSONL into result entries
std::vector<BatchResultEntry> ParseOpenAIBatchResults(const std::string &jsonl_content);

// ============================================================================
// Gemini Batch Operations
// ============================================================================

// Upload JSONL file to Gemini Files API
std::string GeminiUploadBatchFile(ClientContext &context, const std::string &jsonl_content, const std::string &api_key);

// Create a batch job (can use inline or file input)
std::string GeminiCreateBatch(ClientContext &context, const std::string &file_name, const std::string &model,
                              const std::string &api_key);

// Create a batch job with inline requests (for batches <20MB)
std::string GeminiCreateBatchInline(ClientContext &context,
                                    const std::vector<std::pair<std::string, std::string>> &requests,
                                    const std::string &model, const std::string &api_key,
                                    const std::string &display_name, double temperature);

// Get batch status from API
BatchJobInfo GeminiGetBatchStatus(ClientContext &context, const std::string &batch_name, const std::string &api_key);

// Download results file content
std::string GeminiDownloadFile(ClientContext &context, const std::string &file_name, const std::string &api_key);

// Cancel a batch
bool GeminiCancelBatch(ClientContext &context, const std::string &batch_name, const std::string &api_key);

// Parse Gemini results JSONL into result entries
std::vector<BatchResultEntry> ParseGeminiBatchResults(const std::string &jsonl_content);

// Get inline results from completed Gemini batch
std::vector<BatchResultEntry> GeminiGetInlineResults(ClientContext &context, const std::string &batch_name,
                                                     const std::string &api_key);

// ============================================================================
// Function Registration
// ============================================================================

// Register llm_batch_and_cache and llm_hash functions
void RegisterBatchCacheFunctions(ExtensionLoader &loader);

// Register all batch functions
void RegisterBatchFunctions(ExtensionLoader &loader);

} // namespace duckdb
