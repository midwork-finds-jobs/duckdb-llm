#include "providers.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "yyjson.hpp"

namespace duckdb {

using namespace duckdb_yyjson;

std::string EscapeJsonString(const std::string &str) {
	std::string result;
	result.reserve(str.size() + 16);
	for (char c : str) {
		switch (c) {
		case '"':
			result += "\\\"";
			break;
		case '\\':
			result += "\\\\";
			break;
		case '\n':
			result += "\\n";
			break;
		case '\r':
			result += "\\r";
			break;
		case '\t':
			result += "\\t";
			break;
		default:
			if (static_cast<unsigned char>(c) < 32) {
				// Escape control characters
				char buf[8];
				snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
				result += buf;
			} else {
				result += c;
			}
		}
	}
	return result;
}

static std::string WriteDoc(yyjson_mut_doc *doc) {
	char *json = yyjson_mut_write(doc, 0, nullptr);
	if (!json) {
		yyjson_mut_doc_free(doc);
		throw IOException("Failed to serialize JSON");
	}
	std::string result(json);
	free(json);
	yyjson_mut_doc_free(doc);
	return result;
}

std::string BuildOpenAIRequest(const std::string &prompt, const std::string &model, double temperature, int max_tokens,
                               bool json_mode, const std::string &json_schema, const std::string &system_prompt,
                               const std::string &reasoning_effort) {
	auto doc = yyjson_mut_doc_new(nullptr);
	auto root = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, root);

	// model
	yyjson_mut_obj_add_str(doc, root, "model", model.c_str());

	// messages array
	auto messages = yyjson_mut_arr(doc);
	yyjson_mut_obj_add_val(doc, root, "messages", messages);

	// Add system message if provided
	if (!system_prompt.empty()) {
		auto sys_msg = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, sys_msg, "role", "system");
		yyjson_mut_obj_add_str(doc, sys_msg, "content", system_prompt.c_str());
		yyjson_mut_arr_append(messages, sys_msg);
	}

	// Add user message
	auto user_msg = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, user_msg, "role", "user");
	yyjson_mut_obj_add_str(doc, user_msg, "content", prompt.c_str());
	yyjson_mut_arr_append(messages, user_msg);

	// temperature
	yyjson_mut_obj_add_real(doc, root, "temperature", temperature);

	// max_tokens
	yyjson_mut_obj_add_int(doc, root, "max_tokens", max_tokens);

	// response_format for JSON mode
	if (json_mode && json_schema.empty()) {
		auto response_format = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, response_format, "type", "json_object");
		yyjson_mut_obj_add_val(doc, root, "response_format", response_format);
	} else if (!json_schema.empty()) {
		// Structured output with JSON schema
		auto response_format = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, response_format, "type", "json_schema");

		// Parse the provided schema and add it
		yyjson_doc *schema_doc = yyjson_read(json_schema.c_str(), json_schema.size(), 0);
		if (schema_doc) {
			yyjson_val *schema_root = yyjson_doc_get_root(schema_doc);
			auto schema_mut = yyjson_val_mut_copy(doc, schema_root);
			yyjson_mut_obj_add_val(doc, response_format, "json_schema", schema_mut);
			yyjson_doc_free(schema_doc);
		}
		yyjson_mut_obj_add_val(doc, root, "response_format", response_format);
	}

	// reasoning_effort for o1/o3 models
	if (!reasoning_effort.empty()) {
		yyjson_mut_obj_add_str(doc, root, "reasoning_effort", reasoning_effort.c_str());
	}

	return WriteDoc(doc);
}

std::string BuildGeminiRequest(const std::string &prompt, const std::string &model, double temperature, bool json_mode,
                               const std::string &json_schema, const std::string &system_prompt) {
	auto doc = yyjson_mut_doc_new(nullptr);
	auto root = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, root);

	// contents array
	auto contents = yyjson_mut_arr(doc);
	yyjson_mut_obj_add_val(doc, root, "contents", contents);

	// For gemma models, prepend system prompt to user message
	std::string effective_prompt = prompt;
	bool is_gemma = model.find("gemma") != std::string::npos;

	if (is_gemma && !system_prompt.empty()) {
		effective_prompt = system_prompt + "\n\n" + prompt;
	}

	// Add user content
	auto content = yyjson_mut_obj(doc);
	auto parts = yyjson_mut_arr(doc);
	auto text_part = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, text_part, "text", effective_prompt.c_str());
	yyjson_mut_arr_append(parts, text_part);
	yyjson_mut_obj_add_val(doc, content, "parts", parts);
	yyjson_mut_arr_append(contents, content);

	// systemInstruction for gemini-* models (not gemma)
	if (!system_prompt.empty() && !is_gemma) {
		auto sys_instruction = yyjson_mut_obj(doc);
		auto sys_parts = yyjson_mut_arr(doc);
		auto sys_text = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, sys_text, "text", system_prompt.c_str());
		yyjson_mut_arr_append(sys_parts, sys_text);
		yyjson_mut_obj_add_val(doc, sys_instruction, "parts", sys_parts);
		yyjson_mut_obj_add_val(doc, root, "systemInstruction", sys_instruction);
	}

	// generationConfig
	auto gen_config = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_real(doc, gen_config, "temperature", temperature);

	if (json_mode) {
		yyjson_mut_obj_add_str(doc, gen_config, "responseMimeType", "application/json");
	}

	if (!json_schema.empty() && !is_gemma) {
		// Only gemini-* models support responseSchema
		yyjson_doc *schema_doc = yyjson_read(json_schema.c_str(), json_schema.size(), 0);
		if (schema_doc) {
			yyjson_val *schema_root = yyjson_doc_get_root(schema_doc);
			auto schema_mut = yyjson_val_mut_copy(doc, schema_root);
			yyjson_mut_obj_add_val(doc, gen_config, "responseSchema", schema_mut);
			yyjson_doc_free(schema_doc);
		}
	}

	yyjson_mut_obj_add_val(doc, root, "generationConfig", gen_config);

	return WriteDoc(doc);
}

std::string BuildCloudflareRequest(const std::string &prompt, double temperature, int max_tokens,
                                   const std::string &system_prompt) {
	auto doc = yyjson_mut_doc_new(nullptr);
	auto root = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, root);

	// messages array
	auto messages = yyjson_mut_arr(doc);
	yyjson_mut_obj_add_val(doc, root, "messages", messages);

	// Add system message if provided
	if (!system_prompt.empty()) {
		auto sys_msg = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, sys_msg, "role", "system");
		yyjson_mut_obj_add_str(doc, sys_msg, "content", system_prompt.c_str());
		yyjson_mut_arr_append(messages, sys_msg);
	}

	// Add user message
	auto user_msg = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, user_msg, "role", "user");
	yyjson_mut_obj_add_str(doc, user_msg, "content", prompt.c_str());
	yyjson_mut_arr_append(messages, user_msg);

	// temperature and max_tokens
	yyjson_mut_obj_add_real(doc, root, "temperature", temperature);
	yyjson_mut_obj_add_int(doc, root, "max_tokens", max_tokens);

	return WriteDoc(doc);
}

std::string BuildLocalRequest(const std::string &prompt, const std::string &model, double temperature, int max_tokens,
                              bool json_mode, const std::string &system_prompt) {
	// Local uses OpenAI-compatible format
	return BuildOpenAIRequest(prompt, model, temperature, max_tokens, json_mode, "", system_prompt, "");
}

std::string BuildOpenAIUrl(const std::string &base_url) {
	return base_url + "/v1/chat/completions";
}

std::string BuildGeminiUrl(const std::string &base_url, const std::string &model, const std::string &api_key) {
	return base_url + "/v1beta/models/" + model + ":generateContent?key=" + api_key;
}

std::string BuildCloudflareUrl(const std::string &base_url, const std::string &account_id, const std::string &model) {
	return base_url + "/client/v4/accounts/" + account_id + "/ai/run/" + model;
}

std::string BuildLocalUrl(const std::string &base_url) {
	return base_url + "/v1/chat/completions";
}

std::string ParseOpenAIResponse(const std::string &response_body, std::string &error_out) {
	yyjson_doc *doc = yyjson_read(response_body.c_str(), response_body.size(), 0);
	if (!doc) {
		error_out = "Failed to parse response JSON";
		return "";
	}

	yyjson_val *root = yyjson_doc_get_root(doc);

	// Check for error
	yyjson_val *error = yyjson_obj_get(root, "error");
	if (error) {
		yyjson_val *msg = yyjson_obj_get(error, "message");
		if (msg) {
			error_out = yyjson_get_str(msg);
		} else {
			error_out = "Unknown API error";
		}
		yyjson_doc_free(doc);
		return "";
	}

	// Extract choices[0].message.content
	yyjson_val *choices = yyjson_obj_get(root, "choices");
	if (!choices || !yyjson_is_arr(choices)) {
		error_out = "Missing 'choices' in response";
		yyjson_doc_free(doc);
		return "";
	}

	yyjson_val *first_choice = yyjson_arr_get_first(choices);
	if (!first_choice) {
		error_out = "Empty 'choices' array";
		yyjson_doc_free(doc);
		return "";
	}

	yyjson_val *message = yyjson_obj_get(first_choice, "message");
	if (!message) {
		error_out = "Missing 'message' in choice";
		yyjson_doc_free(doc);
		return "";
	}

	yyjson_val *content = yyjson_obj_get(message, "content");
	if (!content) {
		error_out = "Missing 'content' in message";
		yyjson_doc_free(doc);
		return "";
	}

	std::string result = yyjson_get_str(content);
	yyjson_doc_free(doc);
	return result;
}

std::string ParseGeminiResponse(const std::string &response_body, std::string &error_out) {
	yyjson_doc *doc = yyjson_read(response_body.c_str(), response_body.size(), 0);
	if (!doc) {
		error_out = "Failed to parse response JSON";
		return "";
	}

	yyjson_val *root = yyjson_doc_get_root(doc);

	// Check for error
	yyjson_val *error = yyjson_obj_get(root, "error");
	if (error) {
		yyjson_val *msg = yyjson_obj_get(error, "message");
		if (msg) {
			error_out = yyjson_get_str(msg);
		} else {
			error_out = "Unknown API error";
		}
		yyjson_doc_free(doc);
		return "";
	}

	// Extract candidates[0].content.parts[0].text
	yyjson_val *candidates = yyjson_obj_get(root, "candidates");
	if (!candidates || !yyjson_is_arr(candidates)) {
		error_out = "Missing 'candidates' in response";
		yyjson_doc_free(doc);
		return "";
	}

	yyjson_val *first_candidate = yyjson_arr_get_first(candidates);
	if (!first_candidate) {
		error_out = "Empty 'candidates' array";
		yyjson_doc_free(doc);
		return "";
	}

	yyjson_val *content = yyjson_obj_get(first_candidate, "content");
	if (!content) {
		error_out = "Missing 'content' in candidate";
		yyjson_doc_free(doc);
		return "";
	}

	yyjson_val *parts = yyjson_obj_get(content, "parts");
	if (!parts || !yyjson_is_arr(parts)) {
		error_out = "Missing 'parts' in content";
		yyjson_doc_free(doc);
		return "";
	}

	yyjson_val *first_part = yyjson_arr_get_first(parts);
	if (!first_part) {
		error_out = "Empty 'parts' array";
		yyjson_doc_free(doc);
		return "";
	}

	yyjson_val *text = yyjson_obj_get(first_part, "text");
	if (!text) {
		error_out = "Missing 'text' in part";
		yyjson_doc_free(doc);
		return "";
	}

	std::string result = yyjson_get_str(text);
	yyjson_doc_free(doc);
	return result;
}

std::string ParseCloudflareResponse(const std::string &response_body, std::string &error_out) {
	yyjson_doc *doc = yyjson_read(response_body.c_str(), response_body.size(), 0);
	if (!doc) {
		error_out = "Failed to parse response JSON";
		return "";
	}

	yyjson_val *root = yyjson_doc_get_root(doc);

	// Check success flag
	yyjson_val *success = yyjson_obj_get(root, "success");
	if (!success || !yyjson_get_bool(success)) {
		yyjson_val *errors = yyjson_obj_get(root, "errors");
		if (errors && yyjson_is_arr(errors)) {
			yyjson_val *first_error = yyjson_arr_get_first(errors);
			if (first_error) {
				yyjson_val *msg = yyjson_obj_get(first_error, "message");
				if (msg) {
					error_out = yyjson_get_str(msg);
					yyjson_doc_free(doc);
					return "";
				}
			}
		}
		error_out = "API request failed";
		yyjson_doc_free(doc);
		return "";
	}

	// Extract result.response
	yyjson_val *result = yyjson_obj_get(root, "result");
	if (!result) {
		error_out = "Missing 'result' in response";
		yyjson_doc_free(doc);
		return "";
	}

	yyjson_val *response = yyjson_obj_get(result, "response");
	if (!response) {
		error_out = "Missing 'response' in result";
		yyjson_doc_free(doc);
		return "";
	}

	std::string result_str = yyjson_get_str(response);
	yyjson_doc_free(doc);
	return result_str;
}

LlmProvider ParseProvider(const std::string &provider_str) {
	std::string lower = StringUtil::Lower(provider_str);
	if (lower == "local") {
		return LlmProvider::LOCAL;
	} else if (lower == "openai") {
		return LlmProvider::OPENAI;
	} else if (lower == "gemini") {
		return LlmProvider::GEMINI;
	} else if (lower == "cloudflare") {
		return LlmProvider::CLOUDFLARE;
	}
	throw InvalidInputException("Unknown LLM provider: '%s'. Valid providers: local, openai, gemini, cloudflare",
	                            provider_str);
}

std::string ProviderToString(LlmProvider provider) {
	switch (provider) {
	case LlmProvider::LOCAL:
		return "local";
	case LlmProvider::OPENAI:
		return "openai";
	case LlmProvider::GEMINI:
		return "gemini";
	case LlmProvider::CLOUDFLARE:
		return "cloudflare";
	default:
		return "unknown";
	}
}

} // namespace duckdb
