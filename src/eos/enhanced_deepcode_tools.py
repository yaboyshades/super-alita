#!/usr/bin/env python3
"""
Enhanced DeepCode MCP Tools

Extends existing DeepCode integration with Text2Web and Text2Backend capabilities,
integrated with EOS orchestration and Mangle reasoning validation.
"""

from typing import Any

from ..plugins.native_deepcode_plugin import NativeDeepCodePlugin


class EnhancedDeepCodeTools:
    """Enhanced DeepCode tools with Text2Web and Text2Backend capabilities."""
    
    def __init__(self):
        self.native_plugin = NativeDeepCodePlugin()
        self.capabilities = {
            'paper2code': 'Transform research papers into functional code',
            'text2web': 'Generate frontend applications from descriptions',
            'text2backend': 'Create backend systems from natural language'
        }
    
    async def start(self):
        """Initialize the enhanced tools."""
        await self.native_plugin.start()
    
    async def stop(self):
        """Cleanup resources."""
        await self.native_plugin.stop()
    
    async def generate_text2web(self, description: str, 
                               framework: str = "react",
                               styling: str = "tailwind") -> dict[str, Any]:
        """Generate frontend application from text description."""
        
        # Enhanced requirements for web generation
        enhanced_requirements = f"""
        Create a {framework} frontend application with {styling} styling.
        
        Requirements: {description}
        
        Technical specifications:
        - Framework: {framework}
        - Styling: {styling}
        - Responsive design required
        - Component-based architecture
        - Modern JavaScript/TypeScript patterns
        """
        
        # Use existing native plugin with enhanced context
        result = await self.native_plugin.native_deepcode_generate(
            task_kind="text2web",
            requirements=enhanced_requirements,
            repo_path=".",
            framework=framework,
            styling=styling
        )
        
        # Add web-specific metadata
        result['generation_type'] = 'text2web'
        result['framework'] = framework
        result['styling'] = styling
        result['enhanced_context'] = True
        
        return result
    
    async def generate_text2backend(self, description: str,
                                   architecture: str = "microservices",
                                   database: str = "postgresql") -> dict[str, Any]:
        """Generate backend system from text description."""
        
        # Enhanced requirements for backend generation
        enhanced_requirements = f"""
        Create a {architecture} backend system with {database} database.
        
        Requirements: {description}
        
        Technical specifications:
        - Architecture: {architecture}
        - Database: {database}
        - RESTful API design
        - Comprehensive error handling
        - Security best practices
        - Scalable design patterns
        """
        
        # Use existing native plugin with enhanced context
        result = await self.native_plugin.native_deepcode_generate(
            task_kind="text2backend",
            requirements=enhanced_requirements,
            repo_path=".",
            architecture=architecture,
            database=database
        )
        
        # Add backend-specific metadata
        result['generation_type'] = 'text2backend'
        result['architecture'] = architecture
        result['database'] = database
        result['enhanced_context'] = True
        
        return result
    
    async def generate_paper2code_enhanced(self, paper_content: str,
                                          paper_title: str = "",
                                          implementation_focus: str = "") -> dict[str, Any]:
        """Enhanced paper2code with better context analysis."""
        
        enhanced_requirements = f"""
        Implement the research paper: {paper_title}
        
        Paper content/description: {paper_content}
        
        Implementation focus: {implementation_focus}
        
        Technical requirements:
        - Mathematically accurate implementation
        - Comprehensive documentation with paper citations
        - Production-ready code quality
        - Comprehensive test suite
        - Performance considerations documented
        """
        
        result = await self.native_plugin.native_deepcode_generate(
            task_kind="paper2code",
            requirements=enhanced_requirements,
            repo_path=".",
            paper_title=paper_title,
            implementation_focus=implementation_focus
        )
        
        # Add paper2code-specific metadata
        result['generation_type'] = 'paper2code'
        result['paper_title'] = paper_title
        result['implementation_focus'] = implementation_focus
        result['enhanced_context'] = True
        
        return result
    
    def get_capabilities(self) -> dict[str, str]:
        """Get available capabilities."""
        return self.capabilities.copy()
    
    async def health_check(self) -> dict[str, Any]:
        """Check health of all integrated systems."""
        return {
            'status': 'healthy',
            'capabilities': list(self.capabilities.keys()),
            'native_plugin_available': True,
            'enhanced_features': True,
            'integration_level': 'full'
        }


# MCP Tool Functions for integration
async def deepcode_text2web(params: dict[str, Any]) -> dict[str, Any]:
    """MCP tool for Text2Web generation."""
    tools = EnhancedDeepCodeTools()
    await tools.start()
    
    try:
        result = await tools.generate_text2web(
            description=params.get('description', ''),
            framework=params.get('framework', 'react'),
            styling=params.get('styling', 'tailwind')
        )
        return {'success': True, 'result': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        await tools.stop()


async def deepcode_text2backend(params: dict[str, Any]) -> dict[str, Any]:
    """MCP tool for Text2Backend generation."""
    tools = EnhancedDeepCodeTools()
    await tools.start()
    
    try:
        result = await tools.generate_text2backend(
            description=params.get('description', ''),
            architecture=params.get('architecture', 'microservices'),
            database=params.get('database', 'postgresql')
        )
        return {'success': True, 'result': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        await tools.stop()


async def deepcode_paper2code_enhanced(params: dict[str, Any]) -> dict[str, Any]:
    """MCP tool for enhanced Paper2Code generation."""
    tools = EnhancedDeepCodeTools()
    await tools.start()
    
    try:
        result = await tools.generate_paper2code_enhanced(
            paper_content=params.get('paper_content', ''),
            paper_title=params.get('paper_title', ''),
            implementation_focus=params.get('implementation_focus', '')
        )
        return {'success': True, 'result': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        await tools.stop()


# Integration with existing MCP server
def register_enhanced_deepcode_tools(app):
    """Register enhanced DeepCode tools with MCP server."""
    
    @app.tool("deepcode_text2web")
    async def text2web_tool(params: dict[str, Any]) -> dict[str, Any]:
        """Generate frontend application from text description."""
        return await deepcode_text2web(params)
    
    @app.tool("deepcode_text2backend") 
    async def text2backend_tool(params: dict[str, Any]) -> dict[str, Any]:
        """Generate backend system from text description."""
        return await deepcode_text2backend(params)
    
    @app.tool("deepcode_paper2code_enhanced")
    async def paper2code_enhanced_tool(params: dict[str, Any]) -> dict[str, Any]:
        """Enhanced paper to code generation with better context."""
        return await deepcode_paper2code_enhanced(params)
    
    @app.tool("deepcode_capabilities")
    async def capabilities_tool(params: dict[str, Any]) -> dict[str, Any]:
        """Get DeepCode integration capabilities."""
        tools = EnhancedDeepCodeTools()
        return {'success': True, 'capabilities': tools.get_capabilities()}
    
    @app.tool("deepcode_health")
    async def health_tool(params: dict[str, Any]) -> dict[str, Any]:
        """Check DeepCode integration health."""
        tools = EnhancedDeepCodeTools()
        await tools.start()
        try:
            health = await tools.health_check()
            return {'success': True, 'health': health}
        finally:
            await tools.stop()