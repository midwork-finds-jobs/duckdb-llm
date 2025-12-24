#include "llm_batch.hpp"
#include "llm_extension.hpp"
#include "providers.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "yyjson.hpp"
#include <chrono>
#include <mutex>

using namespace duckdb_yyjson; // NOLINT

namespace duckdb {

// Forward declarations
std::string GetLlmConfig(ClientContext &context, const std::string &key);

static std::string GetEnvVar(const char *name) {
	const char *val = getenv(name);
	return val ? std::string(val) : "";
}

// Global sync state - tracks last sync time per provider
static std::mutex sync_mutex;
static std::map<std::string, std::chrono::steady_clock::time_point> last_sync_time;
static const int SYNC_COOLDOWN_SECONDS = 900; // 15 minutes

// ============================================================================
// Provider Detection from Model Name
// ============================================================================

static std::string DetectProviderFromModel(const std::string &model) {
	if (model.find("gemini") != std::string::npos || model.find("gemma") != std::string::npos) {
		return "gemini";
	} else if (model.find("gpt") != std::string::npos || model.find("o1") != std::string::npos ||
	           model.find("o3") != std::string::npos) {
		return "openai";
	} else if (model.find("claude") != std::string::npos) {
		return "anthropic";
	} else if (model.find("llama") != std::string::npos || model.find("@cf/") != std::string::npos) {
		return "cloudflare";
	}
	return "gemini"; // Default
}

// ============================================================================
// Cache Table Management
// ============================================================================

void EnsureBatchCacheTableExists(ClientContext &context) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	conn.Query(R"(
		CREATE TABLE IF NOT EXISTS __llm_cache (
			hash VARCHAR PRIMARY KEY,
			prompt VARCHAR NOT NULL,
			model VARCHAR NOT NULL,
			provider VARCHAR NOT NULL,
			batch_id VARCHAR,
			response VARCHAR,
			status VARCHAR DEFAULT 'pending',
			error_message VARCHAR,
			created_at TIMESTAMP DEFAULT now(),
			updated_at TIMESTAMP DEFAULT now()
		)
	)");
}

// ============================================================================
// Hash Computation
// ============================================================================

static std::string ComputeLlmHash(const std::string &prompt, const std::string &model) {
	std::string provider = DetectProviderFromModel(model);
	std::string input = prompt + "|" + model + "|" + provider;
	hash_t hash_val = Hash(input.c_str(), input.size());
	std::string extended = input + std::to_string(hash_val);
	hash_t hash_val2 = Hash(extended.c_str(), extended.size());
	char hex[33];
	snprintf(hex, sizeof(hex), "%016llx%016llx", (unsigned long long)hash_val, (unsigned long long)hash_val2);
	return std::string(hex);
}

// ============================================================================
// Immediate LLM Call (for llm_and_cache)
// ============================================================================

static std::string ExecuteLlmCall(ClientContext &context, const std::string &prompt, const std::string &model) {
	std::string provider = DetectProviderFromModel(model);
	std::string api_key;
	std::string base_url;

	// Get API key
	if (provider == "gemini") {
		api_key = GetLlmConfig(context, "llm_gemini_api_key");
		if (api_key.empty()) {
			api_key = GetEnvVar("GEMINI_API_KEY");
		}
		base_url = ProviderDefaults::GEMINI_URL;
	} else if (provider == "openai") {
		api_key = GetLlmConfig(context, "llm_openai_api_key");
		if (api_key.empty()) {
			api_key = GetEnvVar("OPENAI_API_KEY");
		}
		base_url = ProviderDefaults::OPENAI_URL;
	} else {
		// Local provider
		base_url = ProviderDefaults::LOCAL_URL;
	}

	// Build request
	std::string request_body;
	std::string url;
	std::string auth_header;
	LlmProvider prov = ParseProvider(provider);

	switch (prov) {
	case LlmProvider::LOCAL:
		request_body = BuildLocalRequest(prompt, model, 0.0, 1024, false, "");
		url = BuildLocalUrl(base_url);
		auth_header = "";
		break;

	case LlmProvider::OPENAI:
		if (api_key.empty()) {
			throw IOException("OpenAI API key required");
		}
		request_body = BuildOpenAIRequest(prompt, model, 0.0, 1024, false, "", "", "");
		url = BuildOpenAIUrl(base_url);
		auth_header = "Bearer " + api_key;
		break;

	case LlmProvider::GEMINI:
		if (api_key.empty()) {
			throw IOException("Gemini API key required");
		}
		request_body = BuildGeminiRequest(prompt, model, 0.0, false, "", "");
		url = BuildGeminiUrl(base_url, model, api_key);
		auth_header = "";
		break;

	default:
		throw IOException("Unsupported provider");
	}

	// Escape for SQL
	std::string escaped_body = StringUtil::Replace(request_body, "'", "''");
	std::string escaped_url = StringUtil::Replace(url, "'", "''");

	// Build query
	std::string query;
	if (auth_header.empty()) {
		query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_post('%s', "
		                           "body := '%s', headers := {'Content-Type': 'application/json'})",
		                           escaped_url, escaped_body);
	} else {
		std::string escaped_auth = StringUtil::Replace(auth_header, "'", "''");
		query =
		    StringUtil::Format("SELECT status, decode(body) AS body FROM http_post('%s', "
		                       "body := '%s', headers := {'Content-Type': 'application/json', 'Authorization': '%s'})",
		                       escaped_url, escaped_body, escaped_auth);
	}

	// Execute
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	auto result = conn.Query(query);
	if (result->HasError()) {
		throw IOException("LLM request failed: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from LLM");
	}

	auto status_code = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).GetValue<std::string>();

	if (status_code < 200 || status_code >= 300) {
		throw IOException("LLM HTTP " + std::to_string(status_code) + ": " + body);
	}

	// Parse response
	std::string error;
	std::string content;

	switch (prov) {
	case LlmProvider::LOCAL:
	case LlmProvider::OPENAI:
		content = ParseOpenAIResponse(body, error);
		break;
	case LlmProvider::GEMINI:
		content = ParseGeminiResponse(body, error);
		break;
	default:
		error = "Unsupported provider";
	}

	if (!error.empty()) {
		throw IOException("LLM error: " + error);
	}

	return content;
}

// ============================================================================
// Background Sync Logic (for batch)
// ============================================================================

static bool ShouldSync(const std::string &provider) {
	std::lock_guard<std::mutex> lock(sync_mutex);
	auto now = std::chrono::steady_clock::now();
	auto it = last_sync_time.find(provider);
	if (it == last_sync_time.end()) {
		return true;
	}
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - it->second).count();
	return elapsed >= SYNC_COOLDOWN_SECONDS;
}

static void MarkSynced(const std::string &provider) {
	std::lock_guard<std::mutex> lock(sync_mutex);
	last_sync_time[provider] = std::chrono::steady_clock::now();
}

static void DoBackgroundSync(ClientContext &context, const std::string &provider, const std::string &model) {
	if (!ShouldSync(provider)) {
		return;
	}

	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	std::string api_key;
	if (provider == "gemini") {
		api_key = GetLlmConfig(context, "llm_gemini_api_key");
		if (api_key.empty()) {
			api_key = GetEnvVar("GEMINI_API_KEY");
		}
	} else if (provider == "openai") {
		api_key = GetLlmConfig(context, "llm_openai_api_key");
		if (api_key.empty()) {
			api_key = GetEnvVar("OPENAI_API_KEY");
		}
	}

	if (api_key.empty()) {
		return;
	}

	MarkSynced(provider);

	try {
		// Submit pending requests
		auto pending_query = StringUtil::Format(
		    "SELECT hash, prompt FROM __llm_cache WHERE status = 'pending' AND provider = '%s' LIMIT 1000", provider);

		auto pending_result = conn.Query(pending_query);
		std::vector<std::pair<std::string, std::string>> requests;

		if (!pending_result->HasError()) {
			while (true) {
				auto chunk = pending_result->Fetch();
				if (!chunk || chunk->size() == 0) {
					break;
				}
				for (idx_t i = 0; i < chunk->size(); i++) {
					requests.push_back({chunk->GetValue(0, i).ToString(), chunk->GetValue(1, i).ToString()});
				}
			}
		}

		if (!requests.empty()) {
			std::string batch_id;
			std::string batch_model = model.empty() ? ProviderDefaults::GEMINI_MODEL : model;

			if (provider == "gemini") {
				batch_id = GeminiCreateBatchInline(context, requests, batch_model, api_key, "llm_auto_batch", 0.0);
			} else if (provider == "openai") {
				if (batch_model.empty()) {
					batch_model = ProviderDefaults::OPENAI_MODEL;
				}
				std::string jsonl = BuildOpenAIBatchJsonl(requests, batch_model, 0.0, 1024, "");
				std::string file_id = OpenAIUploadBatchFile(context, jsonl, api_key, ProviderDefaults::OPENAI_URL);
				batch_id = OpenAICreateBatch(context, file_id, api_key, ProviderDefaults::OPENAI_URL);
			}

			if (!batch_id.empty()) {
				for (const auto &req : requests) {
					auto update = StringUtil::Format(
					    "UPDATE __llm_cache SET status = 'batched', batch_id = '%s', updated_at = now() "
					    "WHERE hash = '%s'",
					    batch_id, req.first);
					conn.Query(update);
				}

				BatchJobInfo job;
				job.batch_id = batch_id;
				job.provider = provider;
				job.model = batch_model;
				job.status = BatchStatus::PENDING;
				job.total_requests = (int)requests.size();
				EnsureBatchTablesExist(context);
				SaveBatchJob(context, job);
			}
		}

		// Sync completed batches
		auto batched_query = StringUtil::Format("SELECT DISTINCT batch_id FROM __llm_cache "
		                                        "WHERE status = 'batched' AND provider = '%s' AND batch_id IS NOT NULL",
		                                        provider);

		auto batched_result = conn.Query(batched_query);
		std::vector<std::string> batch_ids;

		if (!batched_result->HasError()) {
			while (true) {
				auto chunk = batched_result->Fetch();
				if (!chunk || chunk->size() == 0) {
					break;
				}
				for (idx_t i = 0; i < chunk->size(); i++) {
					batch_ids.push_back(chunk->GetValue(0, i).ToString());
				}
			}
		}

		for (const auto &batch_id : batch_ids) {
			try {
				BatchJobInfo status_info;

				if (provider == "gemini") {
					status_info = GeminiGetBatchStatus(context, batch_id, api_key);
				} else if (provider == "openai") {
					status_info = OpenAIGetBatchStatus(context, batch_id, api_key, ProviderDefaults::OPENAI_URL);
				}

				if (status_info.status == BatchStatus::COMPLETED) {
					std::vector<BatchResultEntry> results;

					if (provider == "gemini") {
						results = GeminiGetInlineResults(context, batch_id, api_key);
					} else if (provider == "openai" && !status_info.output_file_id.empty()) {
						std::string content = OpenAIDownloadFile(context, status_info.output_file_id, api_key,
						                                         ProviderDefaults::OPENAI_URL);
						results = ParseOpenAIBatchResults(content);
					}

					for (const auto &entry : results) {
						std::string escaped = StringUtil::Replace(entry.response, "'", "''");
						auto update = StringUtil::Format(
						    "UPDATE __llm_cache SET response = '%s', status = 'completed', updated_at = now() "
						    "WHERE hash = '%s'",
						    escaped, entry.request_id);
						conn.Query(update);
					}
				} else if (status_info.status == BatchStatus::FAILED) {
					auto update = StringUtil::Format(
					    "UPDATE __llm_cache SET status = 'failed', updated_at = now() WHERE batch_id = '%s'", batch_id);
					conn.Query(update);
				}
			} catch (...) {
			}
		}
	} catch (...) {
	}
}

// ============================================================================
// llm_and_cache(prompt, model) -> VARCHAR
// Immediate response with caching
// ============================================================================

struct LlmAndCacheBindData : public FunctionData {
	ClientContext *context;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<LlmAndCacheBindData>();
		copy->context = context;
		return copy;
	}

	bool Equals(const FunctionData &other) const override {
		return true;
	}
};

static unique_ptr<FunctionData> LlmAndCacheBind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	auto bind_data = make_uniq<LlmAndCacheBindData>();
	bind_data->context = &context;
	EnsureBatchCacheTableExists(context);
	return bind_data;
}

static void LlmAndCacheFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<LlmAndCacheBindData>();
	auto &context = *bind_data.context;
	auto &db = DatabaseInstance::GetDatabase(context);

	auto &prompt_vec = args.data[0];
	auto &model_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat prompt_data, model_data;
	prompt_vec.ToUnifiedFormat(count, prompt_data);
	model_vec.ToUnifiedFormat(count, model_data);

	auto prompts = UnifiedVectorFormat::GetData<string_t>(prompt_data);
	auto models = UnifiedVectorFormat::GetData<string_t>(model_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	Connection conn(db);

	for (idx_t i = 0; i < count; i++) {
		auto prompt_idx = prompt_data.sel->get_index(i);
		auto model_idx = model_data.sel->get_index(i);

		if (!prompt_data.validity.RowIsValid(prompt_idx) || !model_data.validity.RowIsValid(model_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		std::string prompt = prompts[prompt_idx].GetString();
		std::string model = models[model_idx].GetString();
		std::string provider = DetectProviderFromModel(model);
		std::string hash = ComputeLlmHash(prompt, model);

		// Check cache first
		auto check =
		    StringUtil::Format("SELECT response FROM __llm_cache WHERE hash = '%s' AND status = 'completed'", hash);
		auto check_result = conn.Query(check);

		if (!check_result->HasError()) {
			auto chunk = check_result->Fetch();
			if (chunk && chunk->size() > 0) {
				auto response_val = chunk->GetValue(0, 0);
				if (!response_val.IsNull()) {
					result_data[i] = StringVector::AddString(result, response_val.ToString());
					continue;
				}
			}
		}

		// Not in cache - call LLM immediately
		try {
			std::string response = ExecuteLlmCall(context, prompt, model);

			// Cache the response
			std::string escaped_prompt = StringUtil::Replace(prompt, "'", "''");
			std::string escaped_response = StringUtil::Replace(response, "'", "''");
			auto insert = StringUtil::Format(
			    "INSERT INTO __llm_cache (hash, prompt, model, provider, response, status) "
			    "VALUES ('%s', '%s', '%s', '%s', '%s', 'completed') "
			    "ON CONFLICT (hash) DO UPDATE SET response = '%s', status = 'completed', updated_at = now()",
			    hash, escaped_prompt, model, provider, escaped_response, escaped_response);
			conn.Query(insert);

			result_data[i] = StringVector::AddString(result, response);
		} catch (std::exception &e) {
			// On error, return NULL
			result_validity.SetInvalid(i);
		}
	}
}

// ============================================================================
// llm_batch_and_cache(prompt, model) -> VARCHAR
// Batch API (50% cheaper) with caching - returns NULL until batch completes
// ============================================================================

struct LlmBatchAndCacheBindData : public FunctionData {
	ClientContext *context;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<LlmBatchAndCacheBindData>();
		copy->context = context;
		return copy;
	}

	bool Equals(const FunctionData &other) const override {
		return true;
	}
};

static unique_ptr<FunctionData> LlmBatchAndCacheBind(ClientContext &context, ScalarFunction &bound_function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	auto bind_data = make_uniq<LlmBatchAndCacheBindData>();
	bind_data->context = &context;
	EnsureBatchCacheTableExists(context);
	return bind_data;
}

static void LlmBatchAndCacheFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<LlmBatchAndCacheBindData>();
	auto &context = *bind_data.context;
	auto &db = DatabaseInstance::GetDatabase(context);

	auto &prompt_vec = args.data[0];
	auto &model_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat prompt_data, model_data;
	prompt_vec.ToUnifiedFormat(count, prompt_data);
	model_vec.ToUnifiedFormat(count, model_data);

	auto prompts = UnifiedVectorFormat::GetData<string_t>(prompt_data);
	auto models = UnifiedVectorFormat::GetData<string_t>(model_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	Connection conn(db);
	std::string last_provider;
	std::string last_model;
	bool has_pending = false;

	for (idx_t i = 0; i < count; i++) {
		auto prompt_idx = prompt_data.sel->get_index(i);
		auto model_idx = model_data.sel->get_index(i);

		if (!prompt_data.validity.RowIsValid(prompt_idx) || !model_data.validity.RowIsValid(model_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		std::string prompt = prompts[prompt_idx].GetString();
		std::string model = models[model_idx].GetString();
		std::string provider = DetectProviderFromModel(model);
		std::string hash = ComputeLlmHash(prompt, model);

		last_provider = provider;
		last_model = model;

		// Check cache
		auto check = StringUtil::Format("SELECT response, status FROM __llm_cache WHERE hash = '%s'", hash);
		auto check_result = conn.Query(check);

		if (!check_result->HasError()) {
			auto chunk = check_result->Fetch();
			if (chunk && chunk->size() > 0) {
				auto response_val = chunk->GetValue(0, 0);
				auto status = chunk->GetValue(1, 0).ToString();

				if (status == "completed" && !response_val.IsNull()) {
					result_data[i] = StringVector::AddString(result, response_val.ToString());
					continue;
				} else {
					result_validity.SetInvalid(i);
					if (status == "pending") {
						has_pending = true;
					}
					continue;
				}
			}
		}

		// Not in cache - insert as pending
		std::string escaped = StringUtil::Replace(prompt, "'", "''");
		auto insert = StringUtil::Format("INSERT INTO __llm_cache (hash, prompt, model, provider, status) "
		                                 "VALUES ('%s', '%s', '%s', '%s', 'pending') ON CONFLICT (hash) DO NOTHING",
		                                 hash, escaped, model, provider);
		conn.Query(insert);
		has_pending = true;
		result_validity.SetInvalid(i);
	}

	// Trigger background sync if we have pending requests
	if (has_pending && !last_provider.empty()) {
		DoBackgroundSync(context, last_provider, last_model);
	}
}

// ============================================================================
// llm_hash(prompt, model) -> VARCHAR
// ============================================================================

static void LlmHashFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &prompt_vec = args.data[0];
	auto &model_vec = args.data[1];
	idx_t count = args.size();

	UnifiedVectorFormat prompt_data, model_data;
	prompt_vec.ToUnifiedFormat(count, prompt_data);
	model_vec.ToUnifiedFormat(count, model_data);

	auto prompts = UnifiedVectorFormat::GetData<string_t>(prompt_data);
	auto models = UnifiedVectorFormat::GetData<string_t>(model_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<string_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto prompt_idx = prompt_data.sel->get_index(i);
		auto model_idx = model_data.sel->get_index(i);

		if (!prompt_data.validity.RowIsValid(prompt_idx) || !model_data.validity.RowIsValid(model_idx)) {
			result_validity.SetInvalid(i);
			continue;
		}

		std::string prompt = prompts[prompt_idx].GetString();
		std::string model = models[model_idx].GetString();
		result_data[i] = StringVector::AddString(result, ComputeLlmHash(prompt, model));
	}
}

// ============================================================================
// Registration
// ============================================================================

void RegisterBatchCacheFunctions(ExtensionLoader &loader) {
	// llm_and_cache(prompt, model) -> VARCHAR (immediate with caching)
	ScalarFunction llm_cache_func("llm_and_cache", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                              LlmAndCacheFunction, LlmAndCacheBind);
	loader.RegisterFunction(llm_cache_func);

	// llm_batch_and_cache(prompt, model) -> VARCHAR (batch API with caching)
	ScalarFunction batch_cache_func("llm_batch_and_cache", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                LogicalType::VARCHAR, LlmBatchAndCacheFunction, LlmBatchAndCacheBind);
	loader.RegisterFunction(batch_cache_func);

	// llm_hash(prompt, model) -> VARCHAR
	ScalarFunction hash_func("llm_hash", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                         LlmHashFunction);
	loader.RegisterFunction(hash_func);
}

} // namespace duckdb
