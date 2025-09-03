import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

export interface SnippetItem extends vscode.QuickPickItem {
    snippet: any;
}

export class SnippetBrowser {
    constructor(private context: vscode.ExtensionContext) {}

    async browseSnippets(): Promise<void> {
        const snippetsPath = path.join(this.context.extensionPath, 'snippets', 'python.json');

        if (!fs.existsSync(snippetsPath)) {
            vscode.window.showErrorMessage('Python snippets not found');
            return;
        }

        try {
            const snippetsContent = fs.readFileSync(snippetsPath, 'utf8');
            const snippets = JSON.parse(snippetsContent);

            // Create quick pick items
            const items: SnippetItem[] = Object.entries(snippets).map(([name, snippet]: [string, any]) => ({
                label: `$(symbol-snippet) ${name}`,
                description: `${snippet.prefix}`,
                detail: snippet.description,
                snippet: snippet
            }));

            // Sort items alphabetically
            items.sort((a, b) => a.label.localeCompare(b.label));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a snippet to insert',
                matchOnDescription: true,
                matchOnDetail: true,
                canPickMany: false
            });

            if (selected) {
                await this.insertSnippet(selected.snippet);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Error loading snippets: ${error}`);
        }
    }

    async browseSnippetsByCategory(): Promise<void> {
        const categories = [
            { label: '$(symbol-class) Super Alita Patterns', category: 'super_alita' },
            { label: '$(symbol-function) Functions & Methods', category: 'functions' },
            { label: '$(symbol-class) Classes & Objects', category: 'classes' },
            { label: '$(symbol-package) Imports & Modules', category: 'imports' },
            { label: '$(symbol-keyword) Control Flow', category: 'control_flow' },
            { label: '$(beaker) Testing & Pytest', category: 'testing' },
            { label: '$(list-unordered) All Snippets', category: 'all' }
        ];

        const selectedCategory = await vscode.window.showQuickPick(categories, {
            placeHolder: 'Select a category to browse',
            canPickMany: false
        });

        if (!selectedCategory) {
            return;
        }

        if (selectedCategory.category === 'all') {
            await this.browseSnippets();
            return;
        }

        const filteredSnippets = await this.getSnippetsByCategory(selectedCategory.category);
        await this.showFilteredSnippets(filteredSnippets, selectedCategory.label);
    }

    async insertSnippet(snippet: any): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const snippetString = new vscode.SnippetString(
            Array.isArray(snippet.body) ? snippet.body.join('\\n') : snippet.body
        );

        await editor.insertSnippet(snippetString);

        // Show information about the inserted snippet
        const message = `Inserted snippet: ${snippet.description || 'Unknown'}`;
        vscode.window.setStatusBarMessage(message, 3000);
    }

    async showSnippetPreview(snippet: any): Promise<void> {
        const content = Array.isArray(snippet.body)
            ? snippet.body.join('\\n')
            : snippet.body;

        const doc = await vscode.workspace.openTextDocument({
            language: 'python',
            content: `# Snippet Preview\\n# Description: ${snippet.description}\\n# Prefix: ${snippet.prefix}\\n\\n${content}`
        });

        await vscode.window.showTextDocument(doc, {
            preview: true,
            viewColumn: vscode.ViewColumn.Beside
        });
    }

    async searchSnippets(): Promise<void> {
        const query = await vscode.window.showInputBox({
            prompt: 'Search snippets by name, prefix, or description',
            placeHolder: 'e.g., function, class, consensus, async...'
        });

        if (!query) {
            return;
        }

        const matchingSnippets = await this.searchSnippetsByQuery(query);
        await this.showFilteredSnippets(matchingSnippets, `Search results for "${query}"`);
    }

    private async getSnippetsByCategory(category: string): Promise<SnippetItem[]> {
        const snippetsPath = path.join(this.context.extensionPath, 'snippets', 'python.json');

        if (!fs.existsSync(snippetsPath)) {
            return [];
        }

        try {
            const snippetsContent = fs.readFileSync(snippetsPath, 'utf8');
            const snippets = JSON.parse(snippetsContent);

            const categoryFilters: { [key: string]: string[] } = {
                'super_alita': ['plugin', 'consensus', 'mcp', 'event', 'super alita'],
                'functions': ['function', 'async', 'lambda', 'def '],
                'classes': ['class', 'subclass', 'mainclass', '__init__'],
                'imports': ['import', 'from'],
                'control_flow': ['if', 'for', 'while', 'try', 'except'],
                'testing': ['pytest', 'test', 'parametrize', '@pytest']
            };

            const keywords = categoryFilters[category] || [];
            const items: SnippetItem[] = [];

            for (const [name, snippet] of Object.entries(snippets) as [string, any][]) {
                const searchText = `${name} ${snippet.description || ''} ${snippet.prefix || ''}`.toLowerCase();

                if (keywords.some(keyword => searchText.includes(keyword.toLowerCase()))) {
                    items.push({
                        label: `$(symbol-snippet) ${name}`,
                        description: `${snippet.prefix}`,
                        detail: snippet.description,
                        snippet: snippet
                    });
                }
            }

            return items.sort((a, b) => a.label.localeCompare(b.label));
        } catch (error) {
            vscode.window.showErrorMessage(`Error filtering snippets: ${error}`);
            return [];
        }
    }

    private async searchSnippetsByQuery(query: string): Promise<SnippetItem[]> {
        const snippetsPath = path.join(this.context.extensionPath, 'snippets', 'python.json');

        if (!fs.existsSync(snippetsPath)) {
            return [];
        }

        try {
            const snippetsContent = fs.readFileSync(snippetsPath, 'utf8');
            const snippets = JSON.parse(snippetsContent);
            const queryLower = query.toLowerCase();
            const items: SnippetItem[] = [];

            for (const [name, snippet] of Object.entries(snippets) as [string, any][]) {
                const searchText = `${name} ${snippet.description || ''} ${snippet.prefix || ''}`.toLowerCase();

                if (searchText.includes(queryLower)) {
                    items.push({
                        label: `$(symbol-snippet) ${name}`,
                        description: `${snippet.prefix}`,
                        detail: snippet.description,
                        snippet: snippet
                    });
                }
            }

            return items.sort((a, b) => a.label.localeCompare(b.label));
        } catch (error) {
            vscode.window.showErrorMessage(`Error searching snippets: ${error}`);
            return [];
        }
    }

    private async showFilteredSnippets(items: SnippetItem[], title: string): Promise<void> {
        if (items.length === 0) {
            vscode.window.showInformationMessage('No snippets found matching the criteria');
            return;
        }

        const actions = [
            { label: '$(insert) Insert Snippet', action: 'insert' },
            { label: '$(eye) Preview Snippet', action: 'preview' }
        ];

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: `${title} (${items.length} snippets found)`,
            matchOnDescription: true,
            matchOnDetail: true,
            canPickMany: false
        });

        if (!selected) {
            return;
        }

        const action = await vscode.window.showQuickPick(actions, {
            placeHolder: 'What would you like to do with this snippet?'
        });

        if (!action) {
            return;
        }

        if (action.action === 'insert') {
            await this.insertSnippet(selected.snippet);
        } else if (action.action === 'preview') {
            await this.showSnippetPreview(selected.snippet);
        }
    }

    async insertSnippetByPrefix(): Promise<void> {
        const prefix = await vscode.window.showInputBox({
            prompt: 'Enter snippet prefix to insert',
            placeHolder: 'e.g., func, alitaplugin, consensus...'
        });

        if (!prefix) {
            return;
        }

        const snippet = await this.findSnippetByPrefix(prefix);
        if (snippet) {
            await this.insertSnippet(snippet);
        } else {
            vscode.window.showInformationMessage(`No snippet found with prefix "${prefix}"`);
        }
    }

    private async findSnippetByPrefix(prefix: string): Promise<any | null> {
        const snippetsPath = path.join(this.context.extensionPath, 'snippets', 'python.json');

        if (!fs.existsSync(snippetsPath)) {
            return null;
        }

        try {
            const snippetsContent = fs.readFileSync(snippetsPath, 'utf8');
            const snippets = JSON.parse(snippetsContent);

            for (const snippet of Object.values(snippets) as any[]) {
                if (snippet.prefix === prefix) {
                    return snippet;
                }
            }

            return null;
        } catch (error) {
            vscode.window.showErrorMessage(`Error finding snippet: ${error}`);
            return null;
        }
    }
}
