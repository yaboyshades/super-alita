#!/usr/bin/env node

/**
 * Test script to verify the Copilot agent endpoint
 * This simulates a basic GitHub Copilot request
 */

const http = require('http');

const testCommands = [
  'health',
  'status', 
  'help',
  'kg machine learning',
  'decide exploration',
  'analyze this code for optimization opportunities'
];

async function testCommand(command) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      copilot_thread_id: 'test_thread_123',
      messages: [{
        role: 'user',
        content: command,
        copilot_references: []
      }],
      stop: null,
      top_p: 1,
      temperature: 0.7,
      max_tokens: 2048,
      presence_penalty: 0,
      frequency_penalty: 0,
      copilot_skills: [],
      agent: 'super-alita'
    });

    const options = {
      hostname: 'localhost',
      port: 8787,
      path: '/copilot',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length,
        // Skip verification for testing (not recommended for production)
        'x-github-copilot-signature': 'test-signature',
        'x-github-copilot-key-id': 'test-key'
      }
    };

    const req = http.request(options, (res) => {
      let response = '';
      
      res.on('data', (chunk) => {
        response += chunk;
      });
      
      res.on('end', () => {
        resolve({
          command,
          statusCode: res.statusCode,
          headers: res.headers,
          body: response
        });
      });
    });

    req.on('error', (err) => {
      reject(err);
    });

    req.write(data);
    req.end();
  });
}

async function runTests() {
  console.log('🧪 Testing Super Alita Copilot Agent...\n');
  
  // Test health endpoint first
  try {
    const healthResponse = await new Promise((resolve, reject) => {
      const req = http.request({
        hostname: 'localhost',
        port: 8787,
        path: '/healthz',
        method: 'GET'
      }, (res) => {
        let response = '';
        res.on('data', (chunk) => response += chunk);
        res.on('end', () => resolve({ statusCode: res.statusCode, body: response }));
      });
      req.on('error', reject);
      req.end();
    });
    
    console.log(`✅ Health check: ${healthResponse.statusCode} - ${healthResponse.body}`);
  } catch (error) {
    console.log(`❌ Health check failed: ${error.message}`);
    console.log('💡 Make sure the server is running: npm start');
    return;
  }
  
  console.log('\n📋 Testing Copilot commands...\n');
  
  for (const command of testCommands) {
    try {
      console.log(`🔄 Testing: "${command}"`);
      const result = await testCommand(command);
      
      console.log(`   Status: ${result.statusCode}`);
      console.log(`   Content-Type: ${result.headers['content-type']}`);
      
      if (result.body.includes('event:') || result.body.includes('data:')) {
        console.log('   ✅ SSE format detected');
        const lines = result.body.split('\n').filter(line => line.trim());
        console.log(`   📄 Response lines: ${lines.length}`);
      } else {
        console.log('   ⚠️  No SSE format detected');
      }
      
      console.log('');
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}\n`);
    }
  }
  
  console.log('🎉 Test suite completed!');
  console.log('\n💡 To see full responses, check server logs or use:');
  console.log('   curl -N -X POST http://localhost:8787/copilot -H "Content-Type: application/json" -d \'{"messages":[{"role":"user","content":"health"}]}\'');
}

if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = { testCommand, runTests };