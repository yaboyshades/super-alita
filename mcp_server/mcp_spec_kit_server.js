// MCP Spec Kit Integration Server (v0.0.47+)
// Bridges Spec Kit SDD phases with Super-Alita neural atoms and world model

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { execSync } from "child_process";
import fs from "fs/promises";
import path from "path";

const server = new McpServer({
  name: "Spec-Kit Enhanced Server",
  version: "2.0.0",
});

// Constitution Management
defineTool(
  "spec-kit-constitution",
  "Create or update project constitution",
  {
    principles_prompt: z.string(),
    project_path: z.string().optional(),
    neural_enhancement: z.boolean().default(true),
    framework_constraints: z.array(z.string()).optional(),
  },
  async ({
    principles_prompt,
    project_path,
    neural_enhancement,
    framework_constraints,
  }) => {
    const projectDir = project_path || process.cwd();
    let enhancedPrinciples = principles_prompt;
    if (neural_enhancement && global.superAlitaIntegration) {
      const enhancement =
        await global.superAlitaIntegration.enhanceConstitution({
          principles: principles_prompt,
          constraints: framework_constraints || [],
          project_context: await loadProjectContext(projectDir),
        });
      enhancedPrinciples = enhancement.enhanced_principles;
    }
    const command = `pwsh scripts/create-constitution.ps1 -PrinciplesPrompt \"${enhancedPrinciples}\"`;
    const result = execSync(command, { cwd: projectDir, encoding: "utf8" });
    return {
      content: [
        {
          type: "text",
          text: `Constitution created successfully:\n\n${result}\n\nEnhanced with neural atoms: ${neural_enhancement}`,
        },
      ],
    };
  }
);

// Specification Creation
defineTool(
  "spec-kit-specify",
  "Create feature specification",
  {
    requirements: z.string(),
    project_path: z.string().optional(),
    enhancement_mode: z
      .enum(["basic", "neural", "world_model"])
      .default("neural"),
    user_personas: z.array(z.string()).optional(),
    success_metrics: z.array(z.string()).optional(),
  },
  async ({
    requirements,
    project_path,
    enhancement_mode,
    user_personas,
    success_metrics,
  }) => {
    const projectDir = project_path || process.cwd();
    let enhancedRequirements = requirements;
    if (enhancement_mode !== "basic" && global.superAlitaIntegration) {
      const enhancement =
        await global.superAlitaIntegration.enhanceSpecification({
          requirements,
          personas: user_personas || [],
          metrics: success_metrics || [],
          mode: enhancement_mode,
        });
      enhancedRequirements = enhancement.enhanced_requirements;
    }
    const command = `pwsh scripts/spec-generation.ps1 -RequirementsPrompt \"${enhancedRequirements}\" -ProjectPath \"${projectDir}\" -FeatureName \"feature\"`;
    const result = execSync(command, { cwd: projectDir, encoding: "utf8" });
    return {
      content: [{ type: "text", text: `Specification created:\n\n${result}` }],
    };
  }
);

// Technical Planning
defineTool(
  "spec-kit-plan",
  "Generate technical implementation plan",
  {
    tech_stack: z.string(),
    project_path: z.string().optional(),
    architecture_style: z
      .enum(["microservices", "monolith", "serverless", "hybrid"])
      .optional(),
    performance_requirements: z
      .object({
        response_time_ms: z.number().optional(),
        concurrent_users: z.number().optional(),
        data_volume: z.string().optional(),
      })
      .optional(),
    integration_requirements: z.array(z.string()).optional(),
  },
  async ({
    tech_stack,
    project_path,
    architecture_style,
    performance_requirements,
    integration_requirements,
  }) => {
    const projectDir = project_path || process.cwd();
    const specPath = path.join(
      projectDir,
      "specs",
      "001-feature-name",
      "spec.md"
    );
    const featurePath = path.dirname(specPath);
    const command = `pwsh scripts/plan-generation.ps1 -TechStackPrompt \"${tech_stack}\" -FeaturePath \"${featurePath}\" -SpecificationPath \"${specPath}\"`;
    const result = execSync(command, { cwd: projectDir, encoding: "utf8" });
    return {
      content: [
        { type: "text", text: `Technical plan generated:\n\n${result}` },
      ],
    };
  }
);

// Task Breakdown
defineTool(
  "spec-kit-tasks",
  "Generate optimized task breakdown",
  {
    project_path: z.string().optional(),
    optimization_goals: z
      .array(
        z.enum(["speed", "quality", "risk_minimization", "parallel_execution"])
      )
      .optional(),
    team_capacity: z
      .object({
        developers: z.number().optional(),
        experience_level: z
          .enum(["junior", "mid", "senior", "mixed"])
          .optional(),
      })
      .optional(),
  },
  async ({ project_path }) => {
    const projectDir = project_path || process.cwd();
    const featurePath = path.join(projectDir, "specs", "001-feature-name");
    const planPath = path.join(featurePath, "plan.md");
    const command = `pwsh scripts/task-orchestration.ps1 -FeaturePath \"${featurePath}\" -PlanPath \"${planPath}\"`;
    const result = execSync(command, { cwd: projectDir, encoding: "utf8" });
    return {
      content: [
        { type: "text", text: `Task breakdown generated:\n\n${result}` },
      ],
    };
  }
);

// Implementation Execution
defineTool(
  "spec-kit-implement",
  "Execute implementation",
  {
    project_path: z.string().optional(),
    dry_run: z.boolean().default(false),
    monitoring_enabled: z.boolean().default(true),
    auto_validation: z.boolean().default(true),
  },
  async ({ project_path, dry_run }) => {
    const projectDir = project_path || process.cwd();
    const featurePath = path.join(projectDir, "specs", "001-feature-name");
    const command = `pwsh scripts/implementation-engine.ps1 -FeaturePath \"${featurePath}\"${
      dry_run ? " -DryRun" : ""
    }`;
    const result = execSync(command, { cwd: projectDir, encoding: "utf8" });
    return {
      content: [
        {
          type: "text",
          text: `Implementation ${
            dry_run ? "preview" : "execution"
          } completed:\n\n${result}`,
        },
      ],
    };
  }
);

// Helper to register tools
function defineTool(name, desc, schema, handler) {
  server.tool(name, desc, schema, handler);
}

// Start the MCP server
const transport = new StdioServerTransport();
await server.connect(transport);
