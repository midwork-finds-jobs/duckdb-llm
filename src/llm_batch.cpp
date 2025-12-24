#include "llm_batch.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "yyjson.hpp"

using namespace duckdb_yyjson; // NOLINT

namespace duckdb {

// ============================================================================
// Status Conversion
// ============================================================================

std::string BatchStatusToString(BatchStatus status) {
	switch (status) {
	case BatchStatus::PENDING:
		return "pending";
	case BatchStatus::VALIDATING:
		return "validating";
	case BatchStatus::IN_PROGRESS:
		return "in_progress";
	case BatchStatus::FINALIZING:
		return "finalizing";
	case BatchStatus::COMPLETED:
		return "completed";
	case BatchStatus::FAILED:
		return "failed";
	case BatchStatus::EXPIRED:
		return "expired";
	case BatchStatus::CANCELLED:
		return "cancelled";
	default:
		return "unknown";
	}
}

BatchStatus ParseOpenAIBatchStatus(const std::string &status) {
	if (status == "validating") {
		return BatchStatus::VALIDATING;
	} else if (status == "in_progress") {
		return BatchStatus::IN_PROGRESS;
	} else if (status == "finalizing") {
		return BatchStatus::FINALIZING;
	} else if (status == "completed") {
		return BatchStatus::COMPLETED;
	} else if (status == "failed") {
		return BatchStatus::FAILED;
	} else if (status == "expired") {
		return BatchStatus::EXPIRED;
	} else if (status == "cancelled") {
		return BatchStatus::CANCELLED;
	}
	return BatchStatus::UNKNOWN;
}

BatchStatus ParseGeminiBatchStatus(const std::string &state) {
	// Handle both JOB_STATE_* (old) and BATCH_STATE_* (current) prefixes
	if (state == "JOB_STATE_PENDING" || state == "BATCH_STATE_PENDING" || state == "PENDING") {
		return BatchStatus::PENDING;
	} else if (state == "JOB_STATE_RUNNING" || state == "BATCH_STATE_RUNNING" || state == "RUNNING") {
		return BatchStatus::IN_PROGRESS;
	} else if (state == "JOB_STATE_SUCCEEDED" || state == "BATCH_STATE_SUCCEEDED" || state == "SUCCEEDED") {
		return BatchStatus::COMPLETED;
	} else if (state == "JOB_STATE_FAILED" || state == "BATCH_STATE_FAILED" || state == "FAILED") {
		return BatchStatus::FAILED;
	} else if (state == "JOB_STATE_CANCELLED" || state == "BATCH_STATE_CANCELLED" || state == "CANCELLED") {
		return BatchStatus::CANCELLED;
	} else if (state == "JOB_STATE_EXPIRED" || state == "BATCH_STATE_EXPIRED" || state == "EXPIRED") {
		return BatchStatus::EXPIRED;
	}
	return BatchStatus::UNKNOWN;
}

// ============================================================================
// Persistence Helpers
// ============================================================================

void EnsureBatchTablesExist(ClientContext &context) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	// Create jobs table
	conn.Query(R"(
		CREATE TABLE IF NOT EXISTS __llm_batch_jobs (
			batch_id VARCHAR PRIMARY KEY,
			provider VARCHAR NOT NULL,
			batch_name VARCHAR,
			model VARCHAR,
			status VARCHAR NOT NULL,
			input_file_id VARCHAR,
			output_file_id VARCHAR,
			error_file_id VARCHAR,
			total_requests INTEGER,
			completed_requests INTEGER DEFAULT 0,
			failed_requests INTEGER DEFAULT 0,
			created_at TIMESTAMP DEFAULT now(),
			updated_at TIMESTAMP DEFAULT now(),
			expires_at TIMESTAMP,
			metadata VARCHAR
		)
	)");

	// Create requests mapping table
	conn.Query(R"(
		CREATE TABLE IF NOT EXISTS __llm_batch_requests (
			batch_id VARCHAR NOT NULL,
			request_id VARCHAR NOT NULL,
			custom_id VARCHAR NOT NULL,
			PRIMARY KEY (batch_id, request_id)
		)
	)");

	// Create results cache table
	conn.Query(R"(
		CREATE TABLE IF NOT EXISTS __llm_batch_results (
			batch_id VARCHAR NOT NULL,
			request_id VARCHAR NOT NULL,
			response VARCHAR,
			status VARCHAR,
			error_message VARCHAR,
			PRIMARY KEY (batch_id, request_id)
		)
	)");
}

void SaveBatchJob(ClientContext &context, const BatchJobInfo &job) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	auto escaped_batch_id = StringUtil::Replace(job.batch_id, "'", "''");
	auto escaped_provider = StringUtil::Replace(job.provider, "'", "''");
	auto escaped_batch_name = StringUtil::Replace(job.batch_name, "'", "''");
	auto escaped_model = StringUtil::Replace(job.model, "'", "''");
	auto escaped_status = BatchStatusToString(job.status);
	auto escaped_input_file = StringUtil::Replace(job.input_file_id, "'", "''");

	auto query = StringUtil::Format(
	    "INSERT INTO __llm_batch_jobs (batch_id, provider, batch_name, model, status, input_file_id, total_requests) "
	    "VALUES ('%s', '%s', '%s', '%s', '%s', '%s', %d)",
	    escaped_batch_id, escaped_provider, escaped_batch_name, escaped_model, escaped_status, escaped_input_file,
	    job.total_requests);

	conn.Query(query);
}

void UpdateBatchJob(ClientContext &context, const BatchJobInfo &job) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	auto escaped_batch_id = StringUtil::Replace(job.batch_id, "'", "''");
	auto escaped_status = BatchStatusToString(job.status);
	auto escaped_output_file = StringUtil::Replace(job.output_file_id, "'", "''");
	auto escaped_error_file = StringUtil::Replace(job.error_file_id, "'", "''");

	auto query = StringUtil::Format("UPDATE __llm_batch_jobs SET "
	                                "status = '%s', "
	                                "output_file_id = '%s', "
	                                "error_file_id = '%s', "
	                                "completed_requests = %d, "
	                                "failed_requests = %d, "
	                                "updated_at = now() "
	                                "WHERE batch_id = '%s'",
	                                escaped_status, escaped_output_file, escaped_error_file, job.completed_requests,
	                                job.failed_requests, escaped_batch_id);

	conn.Query(query);
}

bool GetBatchJob(ClientContext &context, const std::string &batch_id, BatchJobInfo &job) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	auto escaped_id = StringUtil::Replace(batch_id, "'", "''");
	auto query = StringUtil::Format("SELECT batch_id, provider, batch_name, model, status, input_file_id, "
	                                "output_file_id, error_file_id, total_requests, completed_requests, "
	                                "failed_requests FROM __llm_batch_jobs WHERE batch_id = '%s'",
	                                escaped_id);

	auto result = conn.Query(query);
	if (result->HasError() || result->RowCount() == 0) {
		return false;
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		return false;
	}

	job.batch_id = chunk->GetValue(0, 0).ToString();
	job.provider = chunk->GetValue(1, 0).ToString();
	job.batch_name = chunk->GetValue(2, 0).IsNull() ? "" : chunk->GetValue(2, 0).ToString();
	job.model = chunk->GetValue(3, 0).IsNull() ? "" : chunk->GetValue(3, 0).ToString();
	job.status = ParseOpenAIBatchStatus(chunk->GetValue(4, 0).ToString());
	job.input_file_id = chunk->GetValue(5, 0).IsNull() ? "" : chunk->GetValue(5, 0).ToString();
	job.output_file_id = chunk->GetValue(6, 0).IsNull() ? "" : chunk->GetValue(6, 0).ToString();
	job.error_file_id = chunk->GetValue(7, 0).IsNull() ? "" : chunk->GetValue(7, 0).ToString();
	job.total_requests = chunk->GetValue(8, 0).IsNull() ? 0 : chunk->GetValue(8, 0).GetValue<int>();
	job.completed_requests = chunk->GetValue(9, 0).IsNull() ? 0 : chunk->GetValue(9, 0).GetValue<int>();
	job.failed_requests = chunk->GetValue(10, 0).IsNull() ? 0 : chunk->GetValue(10, 0).GetValue<int>();

	return true;
}

void SaveBatchRequests(ClientContext &context, const std::string &batch_id,
                       const std::vector<std::pair<std::string, std::string>> &id_mappings) {
	if (id_mappings.empty()) {
		return;
	}

	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	auto escaped_batch_id = StringUtil::Replace(batch_id, "'", "''");

	// Build bulk insert
	std::string values;
	for (size_t i = 0; i < id_mappings.size(); i++) {
		auto escaped_request_id = StringUtil::Replace(id_mappings[i].first, "'", "''");
		auto escaped_custom_id = StringUtil::Replace(id_mappings[i].second, "'", "''");
		if (i > 0) {
			values += ", ";
		}
		values += StringUtil::Format("('%s', '%s', '%s')", escaped_batch_id, escaped_request_id, escaped_custom_id);
	}

	auto query = "INSERT INTO __llm_batch_requests (batch_id, request_id, custom_id) VALUES " + values;
	conn.Query(query);
}

void CacheBatchResults(ClientContext &context, const std::string &batch_id,
                       const std::vector<BatchResultEntry> &results) {
	if (results.empty()) {
		return;
	}

	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	auto escaped_batch_id = StringUtil::Replace(batch_id, "'", "''");

	for (const auto &entry : results) {
		auto escaped_request_id = StringUtil::Replace(entry.request_id, "'", "''");
		auto escaped_response = StringUtil::Replace(entry.response, "'", "''");
		auto escaped_status = StringUtil::Replace(entry.status, "'", "''");
		auto escaped_error = StringUtil::Replace(entry.error_message, "'", "''");

		auto query = StringUtil::Format(
		    "INSERT OR REPLACE INTO __llm_batch_results (batch_id, request_id, response, status, error_message) "
		    "VALUES ('%s', '%s', '%s', '%s', '%s')",
		    escaped_batch_id, escaped_request_id, escaped_response, escaped_status, escaped_error);
		conn.Query(query);
	}
}

std::vector<BatchResultEntry> GetCachedResults(ClientContext &context, const std::string &batch_id) {
	std::vector<BatchResultEntry> results;

	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	auto escaped_id = StringUtil::Replace(batch_id, "'", "''");
	auto query =
	    StringUtil::Format("SELECT request_id, response, status, error_message FROM __llm_batch_results WHERE batch_id "
	                       "= '%s' ORDER BY request_id",
	                       escaped_id);

	auto result = conn.Query(query);
	if (result->HasError()) {
		return results;
	}

	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}

		for (idx_t i = 0; i < chunk->size(); i++) {
			BatchResultEntry entry;
			entry.request_id = chunk->GetValue(0, i).ToString();
			entry.response = chunk->GetValue(1, i).IsNull() ? "" : chunk->GetValue(1, i).ToString();
			entry.status = chunk->GetValue(2, i).IsNull() ? "ok" : chunk->GetValue(2, i).ToString();
			entry.error_message = chunk->GetValue(3, i).IsNull() ? "" : chunk->GetValue(3, i).ToString();
			results.push_back(entry);
		}
	}

	return results;
}

bool HasCachedResults(ClientContext &context, const std::string &batch_id) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);

	auto escaped_id = StringUtil::Replace(batch_id, "'", "''");
	auto query =
	    StringUtil::Format("SELECT COUNT(*) FROM __llm_batch_results WHERE batch_id = '%s' LIMIT 1", escaped_id);

	auto result = conn.Query(query);
	if (result->HasError()) {
		return false;
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		return false;
	}

	return chunk->GetValue(0, 0).GetValue<int64_t>() > 0;
}

// ============================================================================
// JSONL Builders
// ============================================================================

std::string BuildOpenAIBatchJsonl(const std::vector<std::pair<std::string, std::string>> &requests,
                                  const std::string &model, double temperature, int max_tokens,
                                  const std::string &system_prompt) {
	std::string result;

	for (const auto &req : requests) {
		const auto &request_id = req.first;
		const auto &prompt = req.second;

		yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
		yyjson_mut_val *root = yyjson_mut_obj(doc);
		yyjson_mut_doc_set_root(doc, root);

		// custom_id
		yyjson_mut_obj_add_str(doc, root, "custom_id", request_id.c_str());

		// method
		yyjson_mut_obj_add_str(doc, root, "method", "POST");

		// url
		yyjson_mut_obj_add_str(doc, root, "url", "/v1/chat/completions");

		// body
		yyjson_mut_val *body = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, body, "model", model.c_str());
		yyjson_mut_obj_add_real(doc, body, "temperature", temperature);
		yyjson_mut_obj_add_int(doc, body, "max_tokens", max_tokens);

		// messages array
		yyjson_mut_val *messages = yyjson_mut_arr(doc);

		if (!system_prompt.empty()) {
			yyjson_mut_val *sys_msg = yyjson_mut_obj(doc);
			yyjson_mut_obj_add_str(doc, sys_msg, "role", "system");
			yyjson_mut_obj_add_str(doc, sys_msg, "content", system_prompt.c_str());
			yyjson_mut_arr_append(messages, sys_msg);
		}

		yyjson_mut_val *user_msg = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, user_msg, "role", "user");
		yyjson_mut_obj_add_str(doc, user_msg, "content", prompt.c_str());
		yyjson_mut_arr_append(messages, user_msg);

		yyjson_mut_obj_add_val(doc, body, "messages", messages);
		yyjson_mut_obj_add_val(doc, root, "body", body);

		// Serialize to JSON string
		char *json_str = yyjson_mut_write(doc, 0, nullptr);
		if (json_str) {
			result += json_str;
			result += "\n";
			free(json_str);
		}
		yyjson_mut_doc_free(doc);
	}

	return result;
}

std::string BuildGeminiBatchJsonl(const std::vector<std::pair<std::string, std::string>> &requests,
                                  const std::string &model, double temperature) {
	std::string result;

	for (const auto &req : requests) {
		const auto &request_id = req.first;
		const auto &prompt = req.second;

		yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
		yyjson_mut_val *root = yyjson_mut_obj(doc);
		yyjson_mut_doc_set_root(doc, root);

		// key (request identifier)
		yyjson_mut_obj_add_str(doc, root, "key", request_id.c_str());

		// request object
		yyjson_mut_val *request = yyjson_mut_obj(doc);

		// contents array
		yyjson_mut_val *contents = yyjson_mut_arr(doc);
		yyjson_mut_val *content = yyjson_mut_obj(doc);
		yyjson_mut_val *parts = yyjson_mut_arr(doc);
		yyjson_mut_val *part = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, part, "text", prompt.c_str());
		yyjson_mut_arr_append(parts, part);
		yyjson_mut_obj_add_val(doc, content, "parts", parts);
		yyjson_mut_arr_append(contents, content);
		yyjson_mut_obj_add_val(doc, request, "contents", contents);

		// generationConfig
		yyjson_mut_val *gen_config = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_real(doc, gen_config, "temperature", temperature);
		yyjson_mut_obj_add_val(doc, request, "generationConfig", gen_config);

		yyjson_mut_obj_add_val(doc, root, "request", request);

		// Serialize
		char *json_str = yyjson_mut_write(doc, 0, nullptr);
		if (json_str) {
			result += json_str;
			result += "\n";
			free(json_str);
		}
		yyjson_mut_doc_free(doc);
	}

	return result;
}

// ============================================================================
// OpenAI Batch Operations
// ============================================================================

std::string OpenAIUploadBatchFile(ClientContext &context, const std::string &jsonl_content, const std::string &api_key,
                                  const std::string &base_url) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	// OpenAI Files API requires multipart/form-data
	// http_request extension supports http_post_form for form data
	// We need to send: purpose=batch, file=<content>

	// For now, we'll use a workaround: base64 encode or use direct JSON upload
	// Actually, OpenAI accepts files via multipart - let's check if http_request supports it

	// Fallback: Use direct body with application/json and see if it works
	// Note: This may need adjustment based on http_request capabilities

	auto escaped_content = StringUtil::Replace(jsonl_content, "'", "''");
	auto escaped_key = StringUtil::Replace(api_key, "'", "''");
	auto url = base_url + "/v1/files";

	// Try using http_post_form if available
	auto query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_post_form('%s', "
	                                "params := {'purpose': 'batch', 'file': '%s'}, "
	                                "headers := {'Authorization': 'Bearer %s'})",
	                                url, escaped_content, escaped_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		throw IOException("Failed to upload batch file: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from file upload");
	}

	auto status = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).ToString();

	if (status < 200 || status >= 300) {
		throw IOException("File upload failed with status " + std::to_string(status) + ": " + body);
	}

	// Parse response to get file ID
	yyjson_doc *doc = yyjson_read(body.c_str(), body.size(), 0);
	if (!doc) {
		throw IOException("Failed to parse file upload response");
	}

	yyjson_val *root = yyjson_doc_get_root(doc);
	yyjson_val *id_val = yyjson_obj_get(root, "id");

	std::string file_id;
	if (id_val && yyjson_is_str(id_val)) {
		file_id = yyjson_get_str(id_val);
	}

	yyjson_doc_free(doc);

	if (file_id.empty()) {
		throw IOException("No file ID in upload response: " + body);
	}

	return file_id;
}

std::string OpenAICreateBatch(ClientContext &context, const std::string &file_id, const std::string &api_key,
                              const std::string &base_url) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	// Build request body
	yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
	yyjson_mut_val *root = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, root);

	yyjson_mut_obj_add_str(doc, root, "input_file_id", file_id.c_str());
	yyjson_mut_obj_add_str(doc, root, "endpoint", "/v1/chat/completions");
	yyjson_mut_obj_add_str(doc, root, "completion_window", "24h");

	char *json_str = yyjson_mut_write(doc, 0, nullptr);
	std::string request_body = json_str;
	free(json_str);
	yyjson_mut_doc_free(doc);

	auto escaped_body = StringUtil::Replace(request_body, "'", "''");
	auto escaped_key = StringUtil::Replace(api_key, "'", "''");
	auto url = base_url + "/v1/batches";

	auto query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_post('%s', "
	                                "body := '%s', "
	                                "headers := {'Content-Type': 'application/json', 'Authorization': 'Bearer %s'})",
	                                url, escaped_body, escaped_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		throw IOException("Failed to create batch: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from batch create");
	}

	auto status = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).ToString();

	if (status < 200 || status >= 300) {
		throw IOException("Batch create failed with status " + std::to_string(status) + ": " + body);
	}

	// Parse response
	yyjson_doc *resp_doc = yyjson_read(body.c_str(), body.size(), 0);
	if (!resp_doc) {
		throw IOException("Failed to parse batch create response");
	}

	yyjson_val *resp_root = yyjson_doc_get_root(resp_doc);
	yyjson_val *id_val = yyjson_obj_get(resp_root, "id");

	std::string batch_id;
	if (id_val && yyjson_is_str(id_val)) {
		batch_id = yyjson_get_str(id_val);
	}

	yyjson_doc_free(resp_doc);

	if (batch_id.empty()) {
		throw IOException("No batch ID in create response: " + body);
	}

	return batch_id;
}

BatchJobInfo OpenAIGetBatchStatus(ClientContext &context, const std::string &batch_id, const std::string &api_key,
                                  const std::string &base_url) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	auto escaped_key = StringUtil::Replace(api_key, "'", "''");
	auto url = base_url + "/v1/batches/" + batch_id;

	auto query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_get('%s', "
	                                "headers := {'Authorization': 'Bearer %s'})",
	                                url, escaped_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		throw IOException("Failed to get batch status: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from batch status");
	}

	auto http_status = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).ToString();

	if (http_status < 200 || http_status >= 300) {
		throw IOException("Batch status failed: " + body);
	}

	// Parse response
	BatchJobInfo info;
	info.batch_id = batch_id;
	info.provider = "openai";

	yyjson_doc *doc = yyjson_read(body.c_str(), body.size(), 0);
	if (!doc) {
		throw IOException("Failed to parse batch status response");
	}

	yyjson_val *root = yyjson_doc_get_root(doc);

	yyjson_val *status_val = yyjson_obj_get(root, "status");
	if (status_val && yyjson_is_str(status_val)) {
		info.status = ParseOpenAIBatchStatus(yyjson_get_str(status_val));
	}

	yyjson_val *output_file = yyjson_obj_get(root, "output_file_id");
	if (output_file && yyjson_is_str(output_file)) {
		info.output_file_id = yyjson_get_str(output_file);
	}

	yyjson_val *error_file = yyjson_obj_get(root, "error_file_id");
	if (error_file && yyjson_is_str(error_file)) {
		info.error_file_id = yyjson_get_str(error_file);
	}

	yyjson_val *counts = yyjson_obj_get(root, "request_counts");
	if (counts) {
		yyjson_val *total = yyjson_obj_get(counts, "total");
		yyjson_val *completed = yyjson_obj_get(counts, "completed");
		yyjson_val *failed = yyjson_obj_get(counts, "failed");

		if (total && yyjson_is_int(total)) {
			info.total_requests = (int)yyjson_get_int(total);
		}
		if (completed && yyjson_is_int(completed)) {
			info.completed_requests = (int)yyjson_get_int(completed);
		}
		if (failed && yyjson_is_int(failed)) {
			info.failed_requests = (int)yyjson_get_int(failed);
		}
	}

	yyjson_doc_free(doc);
	return info;
}

std::string OpenAIDownloadFile(ClientContext &context, const std::string &file_id, const std::string &api_key,
                               const std::string &base_url) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	auto escaped_key = StringUtil::Replace(api_key, "'", "''");
	auto url = base_url + "/v1/files/" + file_id + "/content";

	auto query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_get('%s', "
	                                "headers := {'Authorization': 'Bearer %s'})",
	                                url, escaped_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		throw IOException("Failed to download file: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from file download");
	}

	auto status = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).ToString();

	if (status < 200 || status >= 300) {
		throw IOException("File download failed: " + body);
	}

	return body;
}

bool OpenAICancelBatch(ClientContext &context, const std::string &batch_id, const std::string &api_key,
                       const std::string &base_url) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	auto escaped_key = StringUtil::Replace(api_key, "'", "''");
	auto url = base_url + "/v1/batches/" + batch_id + "/cancel";

	auto query = StringUtil::Format("SELECT status FROM http_post('%s', "
	                                "body := '', "
	                                "headers := {'Authorization': 'Bearer %s'})",
	                                url, escaped_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		return false;
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		return false;
	}

	auto status = chunk->GetValue(0, 0).GetValue<int>();
	return status >= 200 && status < 300;
}

std::vector<BatchResultEntry> ParseOpenAIBatchResults(const std::string &jsonl_content) {
	std::vector<BatchResultEntry> results;

	std::istringstream stream(jsonl_content);
	std::string line;

	while (std::getline(stream, line)) {
		if (line.empty()) {
			continue;
		}

		yyjson_doc *doc = yyjson_read(line.c_str(), line.size(), 0);
		if (!doc) {
			continue;
		}

		yyjson_val *root = yyjson_doc_get_root(doc);
		BatchResultEntry entry;

		// Get custom_id
		yyjson_val *custom_id = yyjson_obj_get(root, "custom_id");
		if (custom_id && yyjson_is_str(custom_id)) {
			entry.request_id = yyjson_get_str(custom_id);
		}

		// Check for error
		yyjson_val *error = yyjson_obj_get(root, "error");
		if (error && !yyjson_is_null(error)) {
			entry.status = "error";
			yyjson_val *err_msg = yyjson_obj_get(error, "message");
			if (err_msg && yyjson_is_str(err_msg)) {
				entry.error_message = yyjson_get_str(err_msg);
			}
		} else {
			// Get response
			yyjson_val *response = yyjson_obj_get(root, "response");
			if (response) {
				yyjson_val *body = yyjson_obj_get(response, "body");
				if (body) {
					yyjson_val *choices = yyjson_obj_get(body, "choices");
					if (choices && yyjson_is_arr(choices)) {
						yyjson_val *first = yyjson_arr_get_first(choices);
						if (first) {
							yyjson_val *message = yyjson_obj_get(first, "message");
							if (message) {
								yyjson_val *content = yyjson_obj_get(message, "content");
								if (content && yyjson_is_str(content)) {
									entry.response = yyjson_get_str(content);
									entry.status = "ok";
								}
							}
						}
					}
				}
			}
		}

		yyjson_doc_free(doc);

		if (!entry.request_id.empty()) {
			results.push_back(entry);
		}
	}

	return results;
}

// ============================================================================
// Gemini Batch Operations
// ============================================================================

// Helper to build Gemini inline batch request body
static std::string BuildGeminiInlineBatchBody(const std::vector<std::pair<std::string, std::string>> &requests,
                                              const std::string &display_name, double temperature) {
	yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
	yyjson_mut_val *root = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, root);

	// batch object
	yyjson_mut_val *batch = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, batch, "display_name", display_name.c_str());

	// input_config with inline requests
	yyjson_mut_val *input_config = yyjson_mut_obj(doc);
	yyjson_mut_val *requests_wrapper = yyjson_mut_obj(doc);
	yyjson_mut_val *requests_arr = yyjson_mut_arr(doc);

	for (const auto &req : requests) {
		const auto &request_id = req.first;
		const auto &prompt = req.second;

		yyjson_mut_val *req_obj = yyjson_mut_obj(doc);

		// request object (GenerateContentRequest)
		yyjson_mut_val *request = yyjson_mut_obj(doc);
		yyjson_mut_val *contents = yyjson_mut_arr(doc);
		yyjson_mut_val *content = yyjson_mut_obj(doc);
		yyjson_mut_val *parts = yyjson_mut_arr(doc);
		yyjson_mut_val *part = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, part, "text", prompt.c_str());
		yyjson_mut_arr_append(parts, part);
		yyjson_mut_obj_add_val(doc, content, "parts", parts);
		yyjson_mut_arr_append(contents, content);
		yyjson_mut_obj_add_val(doc, request, "contents", contents);

		// generationConfig
		yyjson_mut_val *gen_config = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_real(doc, gen_config, "temperature", temperature);
		yyjson_mut_obj_add_val(doc, request, "generationConfig", gen_config);

		yyjson_mut_obj_add_val(doc, req_obj, "request", request);

		// metadata with key
		yyjson_mut_val *metadata = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, metadata, "key", request_id.c_str());
		yyjson_mut_obj_add_val(doc, req_obj, "metadata", metadata);

		yyjson_mut_arr_append(requests_arr, req_obj);
	}

	yyjson_mut_obj_add_val(doc, requests_wrapper, "requests", requests_arr);
	yyjson_mut_obj_add_val(doc, input_config, "requests", requests_wrapper);
	yyjson_mut_obj_add_val(doc, batch, "input_config", input_config);
	yyjson_mut_obj_add_val(doc, root, "batch", batch);

	char *json_str = yyjson_mut_write(doc, 0, nullptr);
	std::string result = json_str;
	free(json_str);
	yyjson_mut_doc_free(doc);

	return result;
}

std::string GeminiUploadBatchFile(ClientContext &context, const std::string &jsonl_content,
                                  const std::string &api_key) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	// Gemini uses resumable upload protocol
	// Step 1: Start upload session
	auto num_bytes = std::to_string(jsonl_content.size());
	auto start_url = "https://generativelanguage.googleapis.com/upload/v1beta/files?key=" + api_key;

	// Build metadata request
	yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
	yyjson_mut_val *root = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, root);
	yyjson_mut_val *file = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, file, "display_name", "batch_input");
	yyjson_mut_obj_add_val(doc, root, "file", file);

	char *meta_json = yyjson_mut_write(doc, 0, nullptr);
	std::string metadata_body = meta_json;
	free(meta_json);
	yyjson_mut_doc_free(doc);

	auto escaped_meta = StringUtil::Replace(metadata_body, "'", "''");

	// Start resumable upload
	auto start_query = StringUtil::Format("SELECT status, headers, decode(body) AS body FROM http_post('%s', "
	                                      "body := '%s', "
	                                      "headers := {"
	                                      "'Content-Type': 'application/json', "
	                                      "'X-Goog-Upload-Protocol': 'resumable', "
	                                      "'X-Goog-Upload-Command': 'start', "
	                                      "'X-Goog-Upload-Header-Content-Length': '%s', "
	                                      "'X-Goog-Upload-Header-Content-Type': 'application/jsonl'"
	                                      "})",
	                                      start_url, escaped_meta, num_bytes);

	auto start_result = conn.Query(start_query);
	if (start_result->HasError()) {
		throw IOException("Failed to start Gemini file upload: " + start_result->GetError());
	}

	auto start_chunk = start_result->Fetch();
	if (!start_chunk || start_chunk->size() == 0) {
		throw IOException("No response from Gemini upload start");
	}

	auto start_status = start_chunk->GetValue(0, 0).GetValue<int>();
	if (start_status < 200 || start_status >= 300) {
		auto err_body = start_chunk->GetValue(2, 0).ToString();
		throw IOException("Gemini upload start failed: " + err_body);
	}

	// Parse headers to get upload URL
	auto headers_val = start_chunk->GetValue(1, 0);
	std::string upload_url;

	// Headers come as a MAP type, need to extract x-goog-upload-url
	// For simplicity, let's try a different approach - check response body
	auto resp_body = start_chunk->GetValue(2, 0).ToString();

	// The upload URL is in the response headers, not body
	// http_request returns headers as a struct/map
	// Try parsing as JSON if available
	if (headers_val.type().id() == LogicalTypeId::MAP || headers_val.type().id() == LogicalTypeId::STRUCT) {
		// Convert to string and look for the URL
		auto headers_str = headers_val.ToString();
		// Look for x-goog-upload-url in the headers
		auto url_pos = headers_str.find("x-goog-upload-url");
		if (url_pos != std::string::npos) {
			// Extract the URL value
			auto start_pos = headers_str.find("http", url_pos);
			if (start_pos != std::string::npos) {
				auto end_pos = headers_str.find_first_of("',}", start_pos);
				if (end_pos != std::string::npos) {
					upload_url = headers_str.substr(start_pos, end_pos - start_pos);
				}
			}
		}
	}

	if (upload_url.empty()) {
		throw IOException("Could not extract upload URL from Gemini response");
	}

	// Step 2: Upload actual content
	auto escaped_content = StringUtil::Replace(jsonl_content, "'", "''");
	auto upload_query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_post('%s', "
	                                       "body := '%s', "
	                                       "headers := {"
	                                       "'Content-Length': '%s', "
	                                       "'X-Goog-Upload-Offset': '0', "
	                                       "'X-Goog-Upload-Command': 'upload, finalize'"
	                                       "})",
	                                       upload_url, escaped_content, num_bytes);

	auto upload_result = conn.Query(upload_query);
	if (upload_result->HasError()) {
		throw IOException("Failed to upload Gemini file: " + upload_result->GetError());
	}

	auto upload_chunk = upload_result->Fetch();
	if (!upload_chunk || upload_chunk->size() == 0) {
		throw IOException("No response from Gemini file upload");
	}

	auto upload_status = upload_chunk->GetValue(0, 0).GetValue<int>();
	auto upload_body = upload_chunk->GetValue(1, 0).ToString();

	if (upload_status < 200 || upload_status >= 300) {
		throw IOException("Gemini file upload failed: " + upload_body);
	}

	// Parse response to get file name
	yyjson_doc *resp_doc = yyjson_read(upload_body.c_str(), upload_body.size(), 0);
	if (!resp_doc) {
		throw IOException("Failed to parse Gemini upload response");
	}

	yyjson_val *resp_root = yyjson_doc_get_root(resp_doc);
	yyjson_val *file_obj = yyjson_obj_get(resp_root, "file");
	std::string file_name;

	if (file_obj) {
		yyjson_val *name_val = yyjson_obj_get(file_obj, "name");
		if (name_val && yyjson_is_str(name_val)) {
			file_name = yyjson_get_str(name_val);
		}
	}

	yyjson_doc_free(resp_doc);

	if (file_name.empty()) {
		throw IOException("No file name in Gemini upload response: " + upload_body);
	}

	return file_name;
}

std::string GeminiCreateBatch(ClientContext &context, const std::string &input_source, const std::string &model,
                              const std::string &api_key) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	// Build request body
	yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
	yyjson_mut_val *root = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, root);

	yyjson_mut_val *batch = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, batch, "display_name", "duckdb_batch");

	yyjson_mut_val *input_config = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, input_config, "file_name", input_source.c_str());
	yyjson_mut_obj_add_val(doc, batch, "input_config", input_config);

	yyjson_mut_obj_add_val(doc, root, "batch", batch);

	char *json_str = yyjson_mut_write(doc, 0, nullptr);
	std::string request_body = json_str;
	free(json_str);
	yyjson_mut_doc_free(doc);

	auto escaped_body = StringUtil::Replace(request_body, "'", "''");
	auto url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":batchGenerateContent";

	auto query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_post('%s', "
	                                "body := '%s', "
	                                "headers := {'Content-Type': 'application/json', 'x-goog-api-key': '%s'})",
	                                url, escaped_body, api_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		throw IOException("Failed to create Gemini batch: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from Gemini batch create");
	}

	auto status = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).ToString();

	if (status < 200 || status >= 300) {
		throw IOException("Gemini batch create failed with status " + std::to_string(status) + ": " + body);
	}

	// Parse response to get batch name
	yyjson_doc *resp_doc = yyjson_read(body.c_str(), body.size(), 0);
	if (!resp_doc) {
		throw IOException("Failed to parse Gemini batch create response");
	}

	yyjson_val *resp_root = yyjson_doc_get_root(resp_doc);
	yyjson_val *name_val = yyjson_obj_get(resp_root, "name");

	std::string batch_name;
	if (name_val && yyjson_is_str(name_val)) {
		batch_name = yyjson_get_str(name_val);
	}

	yyjson_doc_free(resp_doc);

	if (batch_name.empty()) {
		throw IOException("No batch name in Gemini create response: " + body);
	}

	return batch_name;
}

// Create batch with inline requests (for smaller batches <20MB)
std::string GeminiCreateBatchInline(ClientContext &context,
                                    const std::vector<std::pair<std::string, std::string>> &requests,
                                    const std::string &model, const std::string &api_key,
                                    const std::string &display_name, double temperature) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	auto request_body = BuildGeminiInlineBatchBody(requests, display_name, temperature);
	auto escaped_body = StringUtil::Replace(request_body, "'", "''");
	auto url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":batchGenerateContent";

	auto query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_post('%s', "
	                                "body := '%s', "
	                                "headers := {'Content-Type': 'application/json', 'x-goog-api-key': '%s'})",
	                                url, escaped_body, api_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		throw IOException("Failed to create Gemini batch: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from Gemini batch create");
	}

	auto status = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).ToString();

	if (status < 200 || status >= 300) {
		throw IOException("Gemini batch create failed with status " + std::to_string(status) + ": " + body);
	}

	// Parse response
	yyjson_doc *resp_doc = yyjson_read(body.c_str(), body.size(), 0);
	if (!resp_doc) {
		throw IOException("Failed to parse Gemini batch create response");
	}

	yyjson_val *resp_root = yyjson_doc_get_root(resp_doc);
	yyjson_val *name_val = yyjson_obj_get(resp_root, "name");

	std::string batch_name;
	if (name_val && yyjson_is_str(name_val)) {
		batch_name = yyjson_get_str(name_val);
	}

	yyjson_doc_free(resp_doc);

	if (batch_name.empty()) {
		throw IOException("No batch name in Gemini create response: " + body);
	}

	return batch_name;
}

BatchJobInfo GeminiGetBatchStatus(ClientContext &context, const std::string &batch_name, const std::string &api_key) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	// batch_name is like "batches/123456"
	auto url = "https://generativelanguage.googleapis.com/v1beta/" + batch_name;

	auto query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_get('%s', "
	                                "headers := {'x-goog-api-key': '%s'})",
	                                url, api_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		throw IOException("Failed to get Gemini batch status: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from Gemini batch status");
	}

	auto http_status = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).ToString();

	if (http_status < 200 || http_status >= 300) {
		throw IOException("Gemini batch status failed: " + body);
	}

	// Parse response
	BatchJobInfo info;
	info.batch_id = batch_name;
	info.provider = "gemini";

	yyjson_doc *doc = yyjson_read(body.c_str(), body.size(), 0);
	if (!doc) {
		throw IOException("Failed to parse Gemini batch status response");
	}

	yyjson_val *root = yyjson_doc_get_root(doc);

	// Get state from metadata or directly
	yyjson_val *metadata = yyjson_obj_get(root, "metadata");
	yyjson_val *state_val = nullptr;
	if (metadata) {
		state_val = yyjson_obj_get(metadata, "state");
	}
	if (!state_val) {
		state_val = yyjson_obj_get(root, "state");
	}

	if (state_val && yyjson_is_str(state_val)) {
		info.status = ParseGeminiBatchStatus(yyjson_get_str(state_val));
	}

	// Get display name
	yyjson_val *display_name = yyjson_obj_get(root, "displayName");
	if (!display_name) {
		display_name = yyjson_obj_get(root, "display_name");
	}
	if (display_name && yyjson_is_str(display_name)) {
		info.batch_name = yyjson_get_str(display_name);
	}

	// Get output file from dest
	yyjson_val *dest = yyjson_obj_get(root, "dest");
	if (dest) {
		yyjson_val *file_name = yyjson_obj_get(dest, "fileName");
		if (!file_name) {
			file_name = yyjson_obj_get(dest, "file_name");
		}
		if (file_name && yyjson_is_str(file_name)) {
			info.output_file_id = yyjson_get_str(file_name);
		}
	}

	yyjson_doc_free(doc);
	return info;
}

std::string GeminiDownloadFile(ClientContext &context, const std::string &file_name, const std::string &api_key) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	// Download file content
	auto url = "https://generativelanguage.googleapis.com/v1beta/" + file_name + ":download";

	auto query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_get('%s', "
	                                "headers := {'x-goog-api-key': '%s'})",
	                                url, api_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		throw IOException("Failed to download Gemini file: " + result->GetError());
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		throw IOException("No response from Gemini file download");
	}

	auto status = chunk->GetValue(0, 0).GetValue<int>();
	auto body = chunk->GetValue(1, 0).ToString();

	if (status < 200 || status >= 300) {
		throw IOException("Gemini file download failed: " + body);
	}

	return body;
}

bool GeminiCancelBatch(ClientContext &context, const std::string &batch_name, const std::string &api_key) {
	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	auto url = "https://generativelanguage.googleapis.com/v1beta/" + batch_name + ":cancel";

	auto query = StringUtil::Format("SELECT status FROM http_post('%s', "
	                                "body := '', "
	                                "headers := {'x-goog-api-key': '%s'})",
	                                url, api_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		return false;
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		return false;
	}

	auto status = chunk->GetValue(0, 0).GetValue<int>();
	return status >= 200 && status < 300;
}

// Get inline results from Gemini batch response
std::vector<BatchResultEntry> GeminiGetInlineResults(ClientContext &context, const std::string &batch_name,
                                                     const std::string &api_key) {
	std::vector<BatchResultEntry> results;

	auto &db = DatabaseInstance::GetDatabase(context);
	Connection conn(db);
	conn.Query("LOAD http_request");

	auto url = "https://generativelanguage.googleapis.com/v1beta/" + batch_name;

	auto query = StringUtil::Format("SELECT status, decode(body) AS body FROM http_get('%s', "
	                                "headers := {'x-goog-api-key': '%s'})",
	                                url, api_key);

	auto result = conn.Query(query);
	if (result->HasError()) {
		return results;
	}

	auto chunk = result->Fetch();
	if (!chunk || chunk->size() == 0) {
		return results;
	}

	auto body = chunk->GetValue(1, 0).ToString();

	yyjson_doc *doc = yyjson_read(body.c_str(), body.size(), 0);
	if (!doc) {
		return results;
	}

	yyjson_val *root = yyjson_doc_get_root(doc);

	// Try multiple paths for inline responses:
	// 1. response.inlinedResponses.inlinedResponses[] (current API)
	// 2. dest.inlinedResponses[] (older API)
	yyjson_val *inlined = nullptr;

	// Path 1: response.inlinedResponses.inlinedResponses
	yyjson_val *response_obj = yyjson_obj_get(root, "response");
	if (response_obj) {
		yyjson_val *inlined_container = yyjson_obj_get(response_obj, "inlinedResponses");
		if (inlined_container) {
			inlined = yyjson_obj_get(inlined_container, "inlinedResponses");
		}
	}

	// Path 2: dest.inlinedResponses (fallback)
	if (!inlined) {
		yyjson_val *dest = yyjson_obj_get(root, "dest");
		if (dest) {
			inlined = yyjson_obj_get(dest, "inlinedResponses");
			if (!inlined) {
				inlined = yyjson_obj_get(dest, "inlined_responses");
			}
		}
	}

	if (inlined && yyjson_is_arr(inlined)) {
		size_t idx, max;
		yyjson_val *item;
		yyjson_arr_foreach(inlined, idx, max, item) {
			BatchResultEntry entry;

			// Get request ID from metadata.key
			yyjson_val *metadata = yyjson_obj_get(item, "metadata");
			if (metadata) {
				yyjson_val *key = yyjson_obj_get(metadata, "key");
				if (key && yyjson_is_str(key)) {
					entry.request_id = yyjson_get_str(key);
				} else {
					entry.request_id = std::to_string(idx);
				}
			} else {
				entry.request_id = std::to_string(idx);
			}

			yyjson_val *response = yyjson_obj_get(item, "response");
			if (response) {
				// Try to get text from response
				yyjson_val *text = yyjson_obj_get(response, "text");
				if (text && yyjson_is_str(text)) {
					entry.response = yyjson_get_str(text);
					entry.status = "ok";
				} else {
					// Try candidates structure: candidates[0].content.parts[0].text
					yyjson_val *candidates = yyjson_obj_get(response, "candidates");
					if (candidates && yyjson_is_arr(candidates)) {
						yyjson_val *first = yyjson_arr_get_first(candidates);
						if (first) {
							yyjson_val *content = yyjson_obj_get(first, "content");
							if (content) {
								yyjson_val *parts = yyjson_obj_get(content, "parts");
								if (parts && yyjson_is_arr(parts)) {
									yyjson_val *part = yyjson_arr_get_first(parts);
									if (part) {
										yyjson_val *txt = yyjson_obj_get(part, "text");
										if (txt && yyjson_is_str(txt)) {
											entry.response = yyjson_get_str(txt);
											entry.status = "ok";
										}
									}
								}
							}
						}
					}
				}
			}

			yyjson_val *error = yyjson_obj_get(item, "error");
			if (error && !yyjson_is_null(error)) {
				entry.status = "error";
				yyjson_val *msg = yyjson_obj_get(error, "message");
				if (msg && yyjson_is_str(msg)) {
					entry.error_message = yyjson_get_str(msg);
				}
			}

			results.push_back(entry);
		}
	}

	yyjson_doc_free(doc);
	return results;
}

std::vector<BatchResultEntry> ParseGeminiBatchResults(const std::string &jsonl_content) {
	std::vector<BatchResultEntry> results;

	std::istringstream stream(jsonl_content);
	std::string line;

	while (std::getline(stream, line)) {
		if (line.empty()) {
			continue;
		}

		yyjson_doc *doc = yyjson_read(line.c_str(), line.size(), 0);
		if (!doc) {
			continue;
		}

		yyjson_val *root = yyjson_doc_get_root(doc);
		BatchResultEntry entry;

		// Get key (request_id)
		yyjson_val *key = yyjson_obj_get(root, "key");
		if (key && yyjson_is_str(key)) {
			entry.request_id = yyjson_get_str(key);
		}

		// Get response
		yyjson_val *response = yyjson_obj_get(root, "response");
		if (response) {
			yyjson_val *candidates = yyjson_obj_get(response, "candidates");
			if (candidates && yyjson_is_arr(candidates)) {
				yyjson_val *first = yyjson_arr_get_first(candidates);
				if (first) {
					yyjson_val *content = yyjson_obj_get(first, "content");
					if (content) {
						yyjson_val *parts = yyjson_obj_get(content, "parts");
						if (parts && yyjson_is_arr(parts)) {
							yyjson_val *part = yyjson_arr_get_first(parts);
							if (part) {
								yyjson_val *text = yyjson_obj_get(part, "text");
								if (text && yyjson_is_str(text)) {
									entry.response = yyjson_get_str(text);
									entry.status = "ok";
								}
							}
						}
					}
				}
			}
		}

		// Check for error
		yyjson_val *error = yyjson_obj_get(root, "error");
		if (error && !yyjson_is_null(error)) {
			entry.status = "error";
			yyjson_val *msg = yyjson_obj_get(error, "message");
			if (msg && yyjson_is_str(msg)) {
				entry.error_message = yyjson_get_str(msg);
			}
		}

		yyjson_doc_free(doc);

		if (!entry.request_id.empty()) {
			results.push_back(entry);
		}
	}

	return results;
}

void RegisterBatchFunctions(ExtensionLoader &loader) {
	RegisterBatchCacheFunctions(loader);
}

} // namespace duckdb
