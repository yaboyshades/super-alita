#!/usr/bin/env node
/**
 * Spec Kit Compiler
 * Compiles specifications to stubs, validators, and tests
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

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

function generatePythonStub(spec) {
    const lines = [
        `# contracts/${spec.spec_id.replace('.', '/')}.pyi`,
        'from typing import TypedDict, Optional',
        ''
    ];
    
    // Add type definitions
    if (spec.io && spec.io.params) {
        for (const param of spec.io.params) {
            if (param.type && param.type.includes('UserIn')) {
                lines.push('class UserIn(TypedDict):');
                lines.push('    email: str');
                lines.push('    name: str');
                lines.push('');
            }
            if (param.type && param.type.includes('UserOut')) {
                lines.push('class UserOut(TypedDict):');
                lines.push('    id: str');
                lines.push('    email: str');
                lines.push('    name: str');
                lines.push('    created_at: str  # ISO8601Z');
                lines.push('');
            }
        }
    }
    
    // Add function signature
    if (spec.io) {
        const params = [];
        if (spec.io.params) {
            for (const param of spec.io.params) {
                let paramStr = param.name;
                if (param.required === false || param.default !== undefined) {
                    paramStr += ': Optional[' + (param.type || 'str') + '] = ...';
                } else {
                    paramStr += ': ' + (param.type || 'str');
                }
                params.push(paramStr);
            }
        }
        const returnType = spec.io.returns || 'None';
        lines.push(`def ${spec.spec_id.split('.').pop()}(${params.join(', ')}) -> ${returnType}: ...`);
    }
    
    return lines.join('\n');
}

function generatePythonValidators(spec) {
    const lines = [
        `# contracts/${spec.spec_id.replace('.', '/')}/validators.py`,
        'import re',
        '',
        'EMAIL_RE = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"',
        ''
    ];
    
    // Pre-conditions
    lines.push(`def pre_${spec.spec_id.split('.').pop()}(data, referrer):`);
    if (spec.contracts && spec.contracts.pre) {
        for (const condition of spec.contracts.pre) {
            if (condition.includes('RFC5322')) {
                lines.push('    assert re.match(EMAIL_RE, data["email"]), "invalid email"');
            } else if (condition.includes('len(data.name) > 0')) {
                lines.push('    assert len(data["name"]) > 0');
            }
        }
    }
    lines.push('');
    
    // Post-conditions
    lines.push(`def post_${spec.spec_id.split('.').pop()}(inp, out):`);
    if (spec.contracts && spec.contracts.post) {
        for (const condition of spec.contracts.post) {
            if (condition.includes('result.email == data.email')) {
                lines.push('    assert out["email"] == inp["data"]["email"]');
            } else if (condition.includes('ISO8601Z')) {
                lines.push('    assert "T" in out["created_at"] and out["created_at"].endswith("Z")');
            } else if (condition.includes('result.id is nonempty')) {
                lines.push('    assert out["id"]');
            }
        }
    }
    lines.push('');
    
    return lines.join('\n');
}

function generatePythonTests(spec) {
    const lines = [
        `# tests/contracts/test_${spec.spec_id.replace('.', '_')}.py`,
        'import pytest',
        'from hypothesis import given, strategies as st',
        `from contracts.${spec.spec_id.replace('.', '.')}.validators import pre_${spec.spec_id.split('.').pop()}, post_${spec.spec_id.split('.').pop()}`,
        `from contracts.${spec.spec_id.replace('.', '.')}.${spec.spec_id.split('.').pop()} import ${spec.spec_id.split('.').pop()}`,
        '',
        '# Hypothesis strategies',
        'def user_in_strategy():',
        '    return st.fixed_dictionaries({',
        '        "email": st.emails(),',
        '        "name": st.text(min_size=1, max_size=80)',
        '    })',
        '',
        '@given(data=user_in_strategy(), referrer=st.none() | st.text(min_size=1, max_size=64))',
        `def test_${spec.spec_id.split('.').pop()}_contract(data, referrer):`,
        `    pre_${spec.spec_id.split('.').pop()}(data, referrer)`,
        `    out = ${spec.spec_id.split('.').pop()}(data, referrer)`,
        `    post_${spec.spec_id.split('.').pop()}({"data": data, "referrer": referrer}, out)`,
        ''
    ];
    
    // Add example-based tests
    if (spec.examples) {
        for (let i = 0; i < spec.examples.length; i++) {
            const example = spec.examples[i];
            lines.push(`def test_${spec.spec_id.split('.').pop()}_example_${example.name || i}():`);
            lines.push(`    # Input: ${JSON.stringify(example.input)}`);
            lines.push(`    # Expected: ${JSON.stringify(example.output)}`);
            lines.push('    pass  # Implementation would go here');
            lines.push('');
        }
    }
    
    return lines.join('\n');
}

function computeHash(content) {
    return crypto.createHash('sha256').update(content, 'utf8').digest('hex');
}

function compileSpec(specFile, outputDir) {
    const content = fs.readFileSync(specFile, 'utf8');
    const spec = parseYaml(content);
    
    // Generate artifacts
    const pyiContent = generatePythonStub(spec);
    const validatorsContent = generatePythonValidators(spec);
    const testsContent = generatePythonTests(spec);
    
    // Create output directories
    const contractDir = path.join(outputDir, 'contracts', spec.spec_id.split('.').slice(0, -1).join('/'));
    const testDir = path.join(outputDir, 'tests', 'contracts');
    
    fs.mkdirSync(contractDir, { recursive: true });
    fs.mkdirSync(testDir, { recursive: true });
    
    // Write artifacts
    const pyiFile = path.join(contractDir, `${spec.spec_id.split('.').pop()}.pyi`);
    const validatorsFile = path.join(contractDir, 'validators.py');
    const testFile = path.join(testDir, `test_${spec.spec_id.replace(/\./g, '_')}.py`);
    
    fs.writeFileSync(pyiFile, pyiContent);
    fs.writeFileSync(validatorsFile, validatorsContent);
    fs.writeFileSync(testFile, testsContent);
    
    // Return metadata for lockfile
    return {
        spec_id: spec.spec_id,
        version: spec.version,
        spec_hash: computeHash(content),
        outputs: {
            pyi: computeHash(pyiContent),
            validators: computeHash(validatorsContent),
            tests: computeHash(testsContent)
        }
    };
}

function main() {
    const args = process.argv.slice(2);
    let specFiles = [];
    let outputDir = '.';
    let lockFile = null;
    
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--out') {
            outputDir = args[++i];
        } else if (args[i] === '--lock') {
            lockFile = args[++i];
        } else if (args[i].endsWith('.yaml') || args[i].endsWith('.yml')) {
            specFiles.push(args[i]);
        }
    }
    
    if (specFiles.length === 0) {
        console.error('No spec files provided');
        process.exit(1);
    }
    
    const results = [];
    for (const specFile of specFiles) {
        if (fs.existsSync(specFile)) {
            try {
                const result = compileSpec(specFile, outputDir);
                results.push(result);
                console.log(`Compiled ${specFile}`);
            } catch (error) {
                console.error(`Error compiling ${specFile}: ${error.message}`);
            }
        } else {
            console.error(`Spec file not found: ${specFile}`);
        }
    }
    
    // Generate lockfile
    if (lockFile) {
        const lockData = {
            specs: results,
            generated_at: new Date().toISOString(),
            signed_by: "spec-bot@ci",
            signature: "placeholder" // In a real implementation, this would be a cryptographic signature
        };
        
        fs.writeFileSync(lockFile, JSON.stringify(lockData, null, 2));
        console.log(`Lockfile written to ${lockFile}`);
    }
    
    console.log(`Compiled ${results.length} specs`);
}

if (require.main === module) {
    main();
}