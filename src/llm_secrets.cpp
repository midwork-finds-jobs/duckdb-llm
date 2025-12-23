#include "llm_extension.hpp"
#include "providers.hpp"
#include "duckdb/main/secret/secret.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

static unique_ptr<BaseSecret> CreateLlmSecret(ClientContext &context, CreateSecretInput &input) {
	auto scope = input.scope;
	if (scope.empty()) {
		scope = {""};
	}

	auto secret = make_uniq<KeyValueSecret>(scope, input.type, input.provider, input.name);

	// Store api_key (redacted in SHOW)
	if (input.options.count("api_key")) {
		secret->secret_map["api_key"] = input.options.at("api_key");
		secret->redact_keys.insert("api_key");
	}

	// Store base_url
	if (input.options.count("base_url")) {
		secret->secret_map["base_url"] = input.options.at("base_url");
	}

	// Store account_id (for Cloudflare)
	if (input.options.count("account_id")) {
		secret->secret_map["account_id"] = input.options.at("account_id");
	}

	return std::move(secret);
}

void RegisterLlmSecrets(ExtensionLoader &loader) {
	// Register the "llm" secret type
	SecretType llm_type;
	llm_type.name = "llm";
	llm_type.deserializer = KeyValueSecret::Deserialize<KeyValueSecret>;
	llm_type.default_provider = "config";
	loader.RegisterSecretType(llm_type);

	// Register create functions for each provider
	const char *providers[] = {"config", "openai", "gemini", "cloudflare", "local"};
	for (const auto &provider : providers) {
		CreateSecretFunction func;
		func.secret_type = "llm";
		func.provider = provider;
		func.function = CreateLlmSecret;
		func.named_parameters["api_key"] = LogicalType::VARCHAR;
		func.named_parameters["base_url"] = LogicalType::VARCHAR;
		func.named_parameters["account_id"] = LogicalType::VARCHAR;
		loader.RegisterFunction(func);
	}
}

} // namespace duckdb
