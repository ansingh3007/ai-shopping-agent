"""
tests/test_shopping_agent.py — Tests for the shopping agent tools.
Run: pytest tests/ -v

These test the tool logic directly against a temporary copy of the database,
so they run without any API key or LLM calls.
"""
import os
import sqlite3
import shutil
import json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Copy the real store.db to a temp location so tests don't mutate it."""
    src = os.path.join(ROOT, "store.db")
    dst = tmp_path / "store.db"
    shutil.copy(src, dst)
    # Point both modules at the temp DB
    import reviews_api
    import shopping_agent
    monkeypatch.setattr(reviews_api, "DB_PATH", str(dst))
    monkeypatch.setattr(shopping_agent, "DB_PATH", str(dst))
    return str(dst)


# ---------------------------------------------------------------------------
# search_products
# ---------------------------------------------------------------------------

class TestSearchProducts:

    def test_search_returns_matching_products(self, temp_db):
        from shopping_agent import search_products
        result = json.loads(search_products.invoke({"query": "honey"}))
        assert len(result) > 0
        assert all("honey" in p["name"].lower() or "honey" in p["description"].lower()
                   or p["category"] == "honey" for p in result)

    def test_search_respects_max_price(self, temp_db):
        from shopping_agent import search_products
        result = json.loads(search_products.invoke({"query": "honey", "max_price": 15.0}))
        assert len(result) > 0
        assert all(p["price"] <= 15.0 for p in result), "max_price filter not applied"

    def test_search_filters_organic_only(self, temp_db):
        from shopping_agent import search_products
        result = json.loads(search_products.invoke({"query": "honey", "is_organic": True}))
        assert len(result) > 0
        assert all(p["is_organic"] is True for p in result), "organic filter not applied"

    def test_search_no_match_returns_empty(self, temp_db):
        from shopping_agent import search_products
        result = json.loads(search_products.invoke({"query": "xyznonexistentproduct"}))
        assert result == []

    def test_search_combined_filters(self, temp_db):
        from shopping_agent import search_products
        result = json.loads(search_products.invoke({
            "query": "honey", "max_price": 20.0, "is_organic": True
        }))
        for p in result:
            assert p["price"] <= 20.0
            assert p["is_organic"] is True


# ---------------------------------------------------------------------------
# get_rating
# ---------------------------------------------------------------------------

class TestGetRating:

    def test_rating_returns_average_and_count(self, temp_db):
        from shopping_agent import get_rating
        result = json.loads(get_rating.invoke({"product_id": 1}))
        assert result["product_id"] == 1
        assert 0 <= result["average_rating"] <= 5
        assert result["review_count"] > 0

    def test_rating_for_product_with_no_reviews(self, temp_db):
        from shopping_agent import get_rating
        result = json.loads(get_rating.invoke({"product_id": 9999}))
        assert result["average_rating"] == 0.0
        assert result["review_count"] == 0


# ---------------------------------------------------------------------------
# checkout
# ---------------------------------------------------------------------------

class TestCheckout:

    def test_checkout_creates_order(self, temp_db):
        from shopping_agent import checkout
        result = checkout.invoke({"product_id": 1})
        assert "confirmed" in result.lower()
        # Verify order persisted
        conn = sqlite3.connect(temp_db)
        count = conn.execute("SELECT COUNT(*) FROM orders WHERE product_id = 1").fetchone()[0]
        conn.close()
        assert count >= 1

    def test_checkout_rejects_invalid_product(self, temp_db):
        from shopping_agent import checkout
        result = checkout.invoke({"product_id": 99999})
        assert "error" in result.lower() or "not found" in result.lower()

    def test_checkout_includes_price_in_confirmation(self, temp_db):
        from shopping_agent import checkout
        # Product 4 is Clover Honey at $8.99
        result = checkout.invoke({"product_id": 4})
        assert "8.99" in result


# ---------------------------------------------------------------------------
# reviews_api (batch)
# ---------------------------------------------------------------------------

class TestReviewsAPI:

    def test_batch_ratings_returns_all_requested(self, temp_db):
        from reviews_api import get_ratings_for_products
        results = get_ratings_for_products([1, 3, 5, 7])
        assert len(results) == 4
        returned_ids = {r["product_id"] for r in results}
        assert returned_ids == {1, 3, 5, 7}

    def test_batch_empty_list_returns_empty(self, temp_db):
        from reviews_api import get_ratings_for_products
        assert get_ratings_for_products([]) == []
