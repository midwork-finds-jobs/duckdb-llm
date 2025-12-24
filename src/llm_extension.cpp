#define DUCKDB_EXTENSION_MAIN

#include "llm_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/connection.hpp"

namespace duckdb {

// Forward declarations for registration functions
void RegisterLlmSecrets(ExtensionLoader &loader);
void RegisterLlmOptions(DatabaseInstance &db);
void RegisterLlmFunction(ExtensionLoader &loader);
void RegisterPromptFunction(ExtensionLoader &loader);
void RegisterBatchFunctions(ExtensionLoader &loader);

static void LoadInternal(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	Connection conn(db);

	// Install and load http_request from community - abort on failure
	auto install_result = conn.Query("INSTALL http_request FROM community");
	if (install_result->HasError()) {
		throw IOException("LLM extension requires http_request extension. Failed to install: " +
		                  install_result->GetError());
	}

	auto load_result = conn.Query("LOAD http_request");
	if (load_result->HasError()) {
		throw IOException("LLM extension requires http_request extension. Failed to load: " + load_result->GetError());
	}

	// Register extension options (SET llm_xxx = ...)
	RegisterLlmOptions(db);

	// Register LLM secret type and create functions
	RegisterLlmSecrets(loader);

	// Register llm() table function
	RegisterLlmFunction(loader);

	// Register prompt() scalar function
	RegisterPromptFunction(loader);

	// Register batch functions
	RegisterBatchFunctions(loader);
}

void LlmExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string LlmExtension::Name() {
	return "llm";
}

std::string LlmExtension::Version() const {
#ifdef EXT_VERSION_LLM
	return EXT_VERSION_LLM;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(llm, loader) {
	duckdb::LoadInternal(loader);
}
}
