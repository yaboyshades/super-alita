#!/usr/bin/env node

/**
 * Simple test script to verify the Copilot agent is working
 * Run this to start the server in development mode
 */

require('dotenv').config();
const server = require('./dist/server.js');

console.log('🚀 Super Alita Copilot Agent starting...');
console.log('📡 Server will be available at http://localhost:8787');
console.log('🔍 Health check: http://localhost:8787/healthz');
console.log('📋 API endpoint: POST http://localhost:8787/copilot');
console.log('');
console.log('💡 To test manually:');
console.log('   curl -N -H "Content-Type: application/json" \\');
console.log('        -X POST http://localhost:8787/copilot \\');
console.log('        -d \'{"messages":[{"role":"user","content":"health"}]}\'');
console.log('');
console.log('🔑 Set GITHUB_TOKEN environment variable for production verification');
console.log('🔌 Set SUPER_ALITA_GRPC_HOST for backend connection (default: localhost:50051)');
console.log('');
console.log('Press Ctrl+C to stop the server');