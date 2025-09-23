#!/usr/bin/env node
/**
 * Spec Kit Lockfile Verifier
 * Verifies that spec lockfiles match the current specs
 */
const fs = require('fs');
const crypto = require('crypto');
const path = require('path');

// Simple YAML parser (in a real implementation, you'd use a proper YAML library)
function parseYaml(content) {
    const lines = content.split('\n');
    const result = {};
    let currentKey = null;
    let currentArray = null;
    
    for (const line of lines) {
        if (line.trim() === '' || line.startsWith('#')) continue;
        
        if (line.startsWith('  - ') && currentArray) {
            // Array item
            const value = line.substring(4).trim();
            if (value.startsWith('"') && value.endsWith('"')) {
                currentArray.push(value.substring(1, value.length - 1));
            } else {
                currentArray.push(value);
            }
        } else if (line.includes(':') && !line.startsWith(' ')) {
            // New top-level key
            const parts = line.split(':');
            const key = parts[0].trim();
            const value = parts.slice(1).join(':').trim();
            
            if (value === '') {
                // Start of array or object
                if (line.endsWith(':')) {
                    currentKey = key;
                    if (line.includes('  - ')) {
                        result[key] = [];
                        currentArray = result[key];
                    } else {
                        result[key] = {};
                    }
                }
            } else {
                // Simple key-value
                if (value.startsWith('"') && value.endsWith('"')) {
                    result[key] = value.substring(1, value.length - 1);
                } else if (value === 'true') {
                    result[key] = true;
                } else if (value === 'false') {
                    result[key] = false;
                } else {
                    result[key] = value;
                }
            }
        }
    }
    
    return result;
}

function computeHash(content) {
    return crypto.createHash('sha256').update(content, 'utf8').digest('hex');
}

function verifySpec(specFile, lockData) {
    if (!fs.existsSync(specFile)) {
        console.error(`Spec file not found: ${specFile}`);
        return false;
    }
    
    const content = fs.readFileSync(specFile, 'utf8');
    const spec = parseYaml(content);
    const specHash = computeHash(content);
    
    // Find matching spec in lockfile
    const lockedSpec = lockData.specs.find(s => s.spec_id === spec.spec_id);
    if (!lockedSpec) {
        console.error(`Spec ${spec.spec_id} not found in lockfile`);
        return false;
    }
    
    if (lockedSpec.spec_hash !== specHash) {
        console.error(`Spec hash mismatch for ${spec.spec_id}`);
        return false;
    }
    
    console.log(`Spec ${spec.spec_id} verified successfully`);
    return true;
}

function main() {
    const lockFile = process.argv[2] || '.contracts/spec.lock.json';
    const specDir = process.argv[3] || 'specs';
    
    if (!fs.existsSync(lockFile)) {
        console.error(`Lockfile not found: ${lockFile}`);
        process.exit(1);
    }
    
    const lockContent = fs.readFileSync(lockFile, 'utf8');
    const lockData = JSON.parse(lockContent);
    
    let allValid = true;
    
    // Verify each spec in the lockfile
    for (const lockedSpec of lockData.specs) {
        // Try to find the spec file
        const specPath = path.join(specDir, `${lockedSpec.spec_id}.yaml`);
        if (!verifySpec(specPath, lockData)) {
            allValid = false;
        }
    }
    
    if (allValid) {
        console.log('All specs verified successfully');
        process.exit(0);
    } else {
        console.error('Spec verification failed');
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}