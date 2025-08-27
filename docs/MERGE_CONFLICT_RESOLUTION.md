# 🔧 Automatic Merge Conflict Resolution

Super Alita includes an intelligent merge conflict resolution system that automatically detects and resolves conflicts in pull requests using advanced strategies.

## ✨ Features

- **🤖 Fully Automated**: Detects conflicts and attempts resolution without manual intervention
- **🧠 Intelligent Strategies**: Uses content analysis to choose the best resolution approach
- **📦 Smart Import Merging**: Automatically combines and deduplicates import statements
- **➕ Additive Change Handling**: Combines non-overlapping additions from both branches
- **💬 Comment Commands**: Manual triggering via GitHub PR comments
- **📊 Comprehensive Reporting**: Detailed analysis and resolution status

## 🚀 How It Works

### Automatic Detection
The workflow automatically triggers when:
- A pull request is opened, synchronized, or reopened
- Merge conflicts are detected between branches

### Resolution Strategies

1. **Auto Strategy** (Default):
   - Analyzes conflict content intelligently
   - Merges import statements automatically
   - Combines additive changes
   - Falls back to manual review for complex conflicts

2. **Current Strategy**:
   - Takes all changes from the current branch
   - Useful when you want to preserve the base branch state

3. **Incoming Strategy**:
   - Takes all changes from the incoming branch
   - Useful when the feature branch should override conflicts

### What Happens Next

1. 🔍 **Detection**: Workflow detects merge conflicts
2. 🤖 **Analysis**: Analyzes each conflicted file
3. 🛠️ **Resolution**: Applies appropriate strategy
4. 📝 **PR Creation**: Creates new PR with resolved conflicts
5. 💬 **Notification**: Comments on original PR with results

## 📋 Usage

### Comment Commands

Use these commands in PR comments to manually trigger resolution:

```bash
# Smart automatic resolution
@github-actions resolve conflicts auto

# Take current branch changes
@github-actions resolve conflicts current

# Take incoming branch changes
@github-actions resolve conflicts incoming
```

### Manual Workflow Dispatch

1. Go to **Actions** → **Automatic Merge Conflict Resolution**
2. Click **"Run workflow"**
3. Enter the PR number
4. Select resolution strategy
5. Click **"Run workflow"**

## 🔍 Example Scenarios

### Smart Import Merging
```python
# Current branch
import os
import sys
import logging

# Incoming branch  
import os
import json
import requests

# Auto-resolved result
import os
import sys
import logging
import json
import requests
```

### Additive Change Combination
```python
# Current branch adds logging
def process_data(data):
    logger.info("Processing data")
    return data

# Incoming branch adds validation
def process_data(data):
    if not data:
        raise ValueError("No data")
    return data

# Auto-resolved combines both
def process_data(data):
    logger.info("Processing data")
    if not data:
        raise ValueError("No data")
    return data
```

## ⚙️ Configuration

The workflow is configured in `.github/workflows/auto-merge-conflict-resolution.yml` and includes:

- **Conflict Detection**: Automatic merge attempt to detect conflicts
- **Multi-Strategy Resolution**: Auto, current, and incoming strategies
- **PR Management**: Automatic creation and updating of resolution PRs
- **Comment Handling**: Parsing and responding to comment commands
- **Error Handling**: Graceful fallback for complex conflicts

## 🛡️ Safety Features

- **Review Required**: All resolutions create PRs that require human review
- **Detailed Reporting**: Complete analysis of what was changed and why
- **Fallback Handling**: Complex conflicts are flagged for manual resolution
- **Git Safety**: All operations preserve original branch history

## 📊 Demo

Run the demo script to see the resolution system in action:

```bash
python demo_merge_conflict_resolution.py
```

This will show how different types of conflicts are analyzed and resolved.

---

*Part of the Super Alita automation system - intelligent development assistance for modern teams.*