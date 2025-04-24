import pytest

from aegiseval.core.regulations import (
    get_regulation_definitions,
    get_regulation_mapping,
    map_test_to_regulations
)


class TestRegulations:
    """Tests for the EU AI Act regulatory mappings."""
    
    def test_get_regulation_definitions(self):
        """Test loading regulation definitions."""
        regulations = get_regulation_definitions()
        
        # Check that core articles are defined
        assert "EU-AI-Act-Article-5" in regulations
        assert "EU-AI-Act-Article-10" in regulations
        assert "EU-AI-Act-Article-13" in regulations
        
        # Check structure of a regulation definition
        article5 = regulations["EU-AI-Act-Article-5"]
        assert "title" in article5
        assert "description" in article5
        assert "article" in article5
        assert "importance" in article5
        assert "url" in article5
    
    def test_get_regulation_mapping(self):
        """Test mapping from categories to regulations."""
        mapping = get_regulation_mapping()
        
        # Check that core categories are mapped
        assert "harmful_instructions" in mapping
        assert "misinformation" in mapping
        assert "bias" in mapping
        
        # Check mappings have correct format
        for category, regs in mapping.items():
            assert isinstance(regs, list)
            assert all(isinstance(r, str) for r in regs)
            assert all(r.startswith("EU-AI-Act-Article-") for r in regs)
    
    def test_map_test_to_regulations(self):
        """Test mapping a test to regulations."""
        # Test with known category
        regs = map_test_to_regulations("harmful_instructions", "test_1")
        assert "EU-AI-Act-Article-5" in regs
        
        # Test with unknown category
        regs = map_test_to_regulations("unknown_category", "test_x")
        assert isinstance(regs, list)
        assert len(regs) == 0 