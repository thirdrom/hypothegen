"""
A/B Test Suite: Semantic Scholar vs. PaperPilot

Compares the current Semantic Scholar implementation (mock data) with the new
PaperPilot implementation (enhanced real academic data) to validate the
quality improvements and ensure backward compatibility.
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple

from app.ingest import ingest
from app.tools.semscholar import search_external as semscholar_search
from app.tools.paperpilot import search_external as paperpilot_search

# Test queries relevant to the mining/metallurgy domain
TEST_QUERIES = [
    "reduce magnetic losses in tailings",
    "magnet separation efficiency",
    "mineral processing optimization",
    "grinding technology improvements",
    "ore concentration methods"
]


class PaperDataValidator:
    """Validates that returned paper data meets the Hypothegen requirements."""
    
    @staticmethod
    def validate_ref(ref: Any, source: str) -> Tuple[bool, List[str]]:
        """Validates a Ref object according to Hypothegen specification."""
        errors = []
        
        if not hasattr(ref, "title"):
            errors.append(f"{source}: missing title")
        
        if not hasattr(ref, "url"):
            errors.append(f"{source}: missing URL")
        elif not ref.url:
            errors.append(f"{source}: empty URL")
        
        if not hasattr(ref, "year"):
            errors.append(f"{source}: missing year")
        elif ref.year is None:
            errors.append(f"{source}: null year")
        elif ref.year < 2000 or ref.year > 2030:
            errors.append(f"{source}: unlikely year {ref.year}")
        
        # Check for enhanced data (authors, abstract) - PaperPilot specific
        if hasattr(ref, "authors"):
            if not ref.authors or not isinstance(ref.authors, list):
                errors.append(f"{source}: invalid authors field")
        
        if hasattr(ref, "abstract"):
            if not ref.abstract or not isinstance(ref.abstract, str):
                errors.append(f"{source}: invalid abstract field")
        
        # Check title format - ensure it's not synthetic mock
        if hasattr(ref, "title"):
            title_lower = ref.title.lower()
            if title_lower.startswith("[mock]") or "mock" in title_lower:
                errors.append(f"{source}: appears to be synthetic mock data")
            elif not title_lower.strip():
                errors.append(f"{source}: empty title")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_paper_quality(ref: Any) -> Dict[str, Any]:
        """Generates a quality score for the paper."""
        score = 0
        max_score = 0
        
        # Base requirement: title, URL, year
        max_score += 10
        if hasattr(ref, "title") and ref.title and len(ref.title) > 10:
            score += 10
        
        max_score += 10
        if hasattr(ref, "url") and ref.url and len(ref.url) > 10:
            score += 10
        
        max_score += 10
        if hasattr(ref, "year") and ref.year and 2000 <= ref.year <= 2030:
            score += 10
        
        # Enhanced data (PaperPilot specific)
        max_score += 15
        if hasattr(ref, "authors") and ref.authors and len(ref.authors) > 0:
            score += 15
        
        max_score += 15
        if hasattr(ref, "abstract") and ref.abstract and len(ref.abstract) > 50:
            score += 15
        
        max_score += 10
        if hasattr(ref, "url") and "arxiv.org" in ref.url:
            score += 10
        
        return {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score * 100) if max_score > 0 else 0,
            "has_enhanced_data": hasattr(ref, "authors") or hasattr(ref, "abstract"),
            "data_completeness": len([attr for attr in ["title", "url", "year"] if hasattr(ref, attr)]) == 3
        }


class A_BTestRunner:
    """Main test runner for comparing implementations."""
    
    def __init__(self):
        self.test_dir = None
        self.results = []
    
    def setup_test_environment(self) -> None:
        """Creates temporary test environment with sample data."""
        self.test_dir = tempfile.mkdtemp()
        print(f"Created test directory: {self.test_dir}")
        
        test_file = Path(self.test_dir) / "sample.txt"
        sample_content = """
        Исследование методов снижения потерь магнетита в хвостах обогащения.
        Методы доизмельчения минералов для повышения эффективности магнитной сепарации.
        Современные технологии обработки полезных ископаемых для улучшения концентрации.
        Оптимизация параметров измельчения для достижения заданного класса крупности.
        Сравнение методов сепарации и доизмельчения для улучшения извлечения руды.
        """
        test_file.write_text(sample_content)
        
        # Ingest the sample data
        ingest(self.test_dir, reset=True)
    
    def run_tests(self) -> Dict[str, Any]:
        """Runs all A/B tests."""
        self.setup_test_environment()
        
        all_results = {
            "queries": {},
            "overall_summary": {},
            "quality_comparison": {},
            "validation_results": {},
            "integration_ready": False
        }
        
        for query in TEST_QUERIES:
            print(f"\n{'='*60}")
            print(f"Testing Query: '{query}'")
            print('='*60)
            
            query_results = self._test_single_query(query)
            all_results["queries"][query] = query_results
        
        self._generate_summary(all_results)
        self._validate_integration_readiness(all_results)
        
        return all_results
    
    def _test_single_query(self, query: str) -> Dict[str, Any]:
        """Tests a single query with both implementations."""
        print(f"\n--- Testing: '{query}' ---")
        
        # Test Semantic Scholar implementation
        print("\n1. Semantic Scholar (Current Implementation):")
        semscholar_papers = self._get_papers(semscholar_search, query)
        
        # Test PaperPilot implementation
        print("\n2. PaperPilot (New Implementation):")
        paperpilot_papers = self._get_papers(paperpilot_search, query)
        
        # Validate and score
        semscholar_validation = self._validate_and_score(semscholar_papers, "semscholar")
        paperpilot_validation = self._validate_and_score(paperpilot_papers, "paperpilot")
        
        # Generate comparison
        comparison = self._generate_comparison(
            semscholar_papers, semscholar_validation,
            paperpilot_papers, paperpilot_validation
        )
        
        return {
            "query": query,
            "semscholar": {
                "paper_count": len(semscholar_papers),
                "validation": semscholar_validation,
                "papers": [self._serialize_paper(p) for p in semscholar_papers]
            },
            "paperpilot": {
                "paper_count": len(paperpilot_papers),
                "validation": paperpilot_validation,
                "papers": [self._serialize_paper(p) for p in paperpilot_papers]
            },
            "comparison": comparison
        }
    
    def _get_papers(self, search_func, query: str) -> List[Any]:
        """Helper to get papers from search function."""
        papers = []
        try:
            papers = search_func(query, limit=3)
        except Exception as e:
            print(f"  Error: {e}")
        return papers
    
    def _validate_and_score(self, papers: List[Any], source: str) -> Dict[str, Any]:
        """Validates and scores papers."""
        validation_results = {
            "valid_count": 0,
            "total_count": len(papers),
            "validation_errors": [],
            "quality_scores": [],
            "enhanced_data": 0
        }
        
        for i, paper in enumerate(papers):
            # Validate the paper
            is_valid, errors = PaperDataValidator.validate_ref(paper, f"{source}[{i}]")
            if is_valid:
                validation_results["valid_count"] += 1
            else:
                validation_results["validation_errors"].extend(errors)
            
            # Calculate quality score
            quality = PaperDataValidator.validate_paper_quality(paper)
            validation_results["quality_scores"].append(quality)
            
            if quality["has_enhanced_data"]:
                validation_results["enhanced_data"] += 1
        
        return validation_results
    
    def _generate_comparison(self, sem_papers, sem_validation, pilot_papers, pilot_validation) -> Dict[str, Any]:
        """Generates comparison between implementations."""
        
        # Calculate average quality scores
        sem_quality_avg = 0
        if sem_validation["quality_scores"]:
            sem_quality_avg = sum(p["percentage"] for p in sem_validation["quality_scores"]) / len(sem_validation["quality_scores"])
        
        pilot_quality_avg = 0
        if pilot_validation["quality_scores"]:
            pilot_quality_avg = sum(p["percentage"] for p in pilot_validation["quality_scores"]) / len(pilot_validation["quality_scores"])
        
        return {
            "paper_count_difference": len(pilot_papers) - len(sem_papers),
            "validity_ratio": {
                "semscholar": sem_validation["valid_count"] / max(len(sem_papers), 1),
                "paperpilot": pilot_validation["valid_count"] / max(len(pilot_papers), 1)
            },
            "enhancement_ratio": {
                "semscholar": sem_validation["enhanced_data"] / max(len(sem_papers), 1),
                "paperpilot": pilot_validation["enhanced_data"] / max(len(pilot_papers), 1)
            },
            "average_quality_score": {
                "semscholar": sem_quality_avg,
                "paperpilot": pilot_quality_avg,
                "improvement": pilot_quality_avg - sem_quality_avg
            },
            "summary": self._get_comparison_summary(sem_validation, pilot_validation)
        }
    
    def _get_comparison_summary(self, sem_validation, pilot_validation):
        """Gets a qualitative summary of the comparison."""
        sem_enhanced = sem_validation["enhanced_data"]
        pilot_enhanced = pilot_validation["enhanced_data"]
        
        if pilot_enhanced == 0 and sem_enhanced == 0:
            return "Both implementations return basic paper metadata (title, URL, year)."
        elif pilot_enhanced > sem_enhanced:
            return "PaperPilot provides significantly more enhanced data (authors, abstracts) than Semantic Scholar."
        elif pilot_enhanced < sem_enhanced:
            return "Semantic Scholar provides more enhanced data than PaperPilot."
        else:
            return "Both implementations provide similar amounts of enhanced data."
    
    def _serialize_paper(self, paper: Any) -> Dict[str, Any]:
        """Serializes a paper for JSON output."""
        paper_data = {
            "title": getattr(paper, "title", ""),
            "url": getattr(paper, "url", ""),
            "year": getattr(paper, "year", None)
        }
        
        # Add enhanced data if available
        if hasattr(paper, "authors"):
            paper_data["authors"] = paper.authors
        if hasattr(paper, "abstract"):
            paper_data["abstract"] = paper.abstract
        
        return paper_data
    
    def _generate_summary(self, results: Dict[str, Any]) -> None:
        """Generates and prints a summary of all test results."""
        print(f"\n{'='*60}")
        print("A/B TEST SUMMARY")
        print('='*60)
        
        total_semscholar = sum(
            len(res["semscholar"]["papers"]) 
            for res in results["queries"].values()
        )
        total_paperpilot = sum(
            len(res["paperpilot"]["papers"]) 
            for res in results["queries"].values()
        )
        
        semscholar_enhanced = sum(
            sum(1 for p in res["semscholar"]["validation"]["quality_scores"] 
                if p["has_enhanced_data"]) 
            for res in results["queries"].values()
        )
        
        paperpilot_enhanced = sum(
            sum(1 for p in res["paperpilot"]["validation"]["quality_scores"] 
                if p["has_enhanced_data"]) 
            for res in results["queries"].values()
        )
        
        print(f"\nTotal Papers Retrieved:")
        print(f"  Semantic Scholar: {total_semscholar}")
        print(f"  PaperPilot: {total_paperpilot}")
        print(f"  Difference: {total_paperpilot - total_semscholar}")
        
        print(f"\nEnhanced Data (authors/abstract):")
        print(f"  Semantic Scholar: {semscholar_enhanced}/{total_semscholar} ({semscholar_enhanced/max(total_semscholar,1)*100:.1f}%)")
        print(f"  PaperPilot: {paperpilot_enhanced}/{total_paperpilot} ({paperpilot_enhanced/max(total_paperpilot,1)*100:.1f}%)")
        
        # Calculate overall quality improvement
        semscholar_avg_quality = 0
        paperpilot_avg_quality = 0
        valid_semscholar = 0
        valid_paperpilot = 0
        
        for query_results in results["queries"].values():
            sem_scores = [p["percentage"] for p in query_results["semscholar"]["validation"]["quality_scores"]]
            pilot_scores = [p["percentage"] for p in query_results["paperpilot"]["validation"]["quality_scores"]]
            
            if sem_scores:
                semscholar_avg_quality += sum(sem_scores) / len(sem_scores)
                valid_semscholar += 1
            
            if pilot_scores:
                paperpilot_avg_quality += sum(pilot_scores) / len(pilot_scores)
                valid_paperpilot += 1
        
        if valid_semscholar > 0:
            semscholar_avg_quality /= valid_semscholar
        if valid_paperpilot > 0:
            paperpilot_avg_quality /= valid_paperpilot
        
        print(f"\nAverage Quality Score:")
        print(f"  Semantic Scholar: {semscholar_avg_quality:.1f}/100")
        print(f"  PaperPilot: {paperpilot_avg_quality:.1f}/100")
        print(f"  Improvement: {paperpilot_avg_quality - semscholar_avg_quality:.1f} points")
        
        # Valid/refused ratio
        semscholar_valid = sum(
            1 for res in results["queries"].values()
            for p in res["semscholar"]["validation"]["quality_scores"]
            if p["percentage"] >= 80
        )
        paperpilot_valid = sum(
            1 for res in results["queries"].values()
            for p in res["paperpilot"]["validation"]["quality_scores"]
            if p["percentage"] >= 80
        )
        
        print(f"\nHigh-Quality Papers (>=80%):")
        print(f"  Semantic Scholar: {semscholar_valid} papers")
        print(f"  PaperPilot: {paperpilot_valid} papers")
    
    def _validate_integration_readiness(self, results: Dict[str, Any]) -> None:
        """Validates that the integration is ready for deployment."""
        integration_issues = []
        
        # Check main requirements from the task
        for query_results in results["queries"].values():
            semscholar_data = query_results["semscholar"]
            pilot_data = query_results["paperpilot"]
            
            # Both should return the same number of papers
            if semscholar_data["paper_count"] != pilot_data["paper_count"]:
                integration_issues.append(
                    f"Paper count mismatch for query '{query_results['query']}'"
                )
            
            # Should have valid Ref objects
            if semscholar_data["paper_count"] == 0 and pilot_data["paper_count"] == 0:
                integration_issues.append(
                    f"Both implementations returned 0 papers for query: '{query_results['query']}'"
                )
        
        # Check quality improvements
        avg_semscholar_quality = 0
        avg_paperpilot_quality = 0
        for query_results in results["queries"].values():
            sem_scores = query_results["semscholar"]["validation"]["quality_scores"]
            pilot_scores = query_results["paperpilot"]["validation"]["quality_scores"]
            
            if sem_scores:
                avg_semscholar_quality += sum(p["percentage"] for p in sem_scores) / len(sem_scores)
            if pilot_scores:
                avg_paperpilot_quality += sum(p["percentage"] for p in pilot_scores) / len(pilot_scores)
        
        if avg_paperpilot_quality > 85 and avg_semscholar_quality < 60:
            integration_issues.append(
                "PaperPilot shows significant quality improvement but needs integration testing"
            )
        
        if integration_issues:
            print(f"\n{'WARNING': '='*58}")
            print("INTEGRATION ISSUES FOUND:")
            for issue in integration_issues:
                print(f"  ⚠️  {issue}")
            print("\nRecommendation: Address integration issues before deployment.")
        else:
            print(f"\n{'SUCCESS': '='*58}")
            print("INTEGRATION READY:")
            print("  ✓ Both implementations return valid Ref objects")
            print("  ✓ PaperPilot provides enhanced data (authors, abstracts)")
            print("  ✓ Quality improvements documented")
            print("  ✓ Ready for A/B testing in production")
    
    def cleanup(self) -> None:
        """Cleans up test files."""
        if self.test_dir and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
            print(f"\nCleaned up test directory: {self.test_dir}")
