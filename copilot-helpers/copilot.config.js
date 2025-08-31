module.exports = {
  // Define common prompt scaffolds for Super Alita
  templates: {
    askExpert: "You are an expert {role} specializing in Super Alita development. \"{input}\". Provide {format}. Steps:",
    contextBlock: "=== CONTEXT ===\n• {title}: {snippet}\n=== END ===\n",
    deepConf: "Use DeepConf consensus with method: {method}, num_samples: {samples}, temperature: {temp}",
    reugTemplate: "REUG streaming orchestration with {eventType} events for {purpose}"
  },

  // Register custom snippets for Copilot
  snippets: {
    "optimize-prompt": {
      "prefix": "opt-prompt",
      "body": [
        "${templates.askExpert.replace('{role}', '$1').replace('{input}', '$2').replace('{format}', '$3')}",
        "Constraints: $4",
        "Let's think step by step…"
      ]
    },
    "inject-context": {
      "prefix": "ctx-block",
      "body": [
        "${templates.contextBlock.replace('{title}', '$1').replace('{snippet}', '$2')}"
      ]
    },
    "deepconf-consensus": {
      "prefix": "deep-conf",
      "body": [
        "${templates.deepConf.replace('{method}', '$1').replace('{samples}', '$2').replace('{temp}', '$3')}"
      ]
    },
    "reug-stream": {
      "prefix": "reug",
      "body": [
        "${templates.reugTemplate.replace('{eventType}', '$1').replace('{purpose}', '$2')}"
      ]
    }
  }
};
