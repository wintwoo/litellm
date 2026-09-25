"""
Unit tests for Vertex AI Dedicated Endpoint support

Tests that LiteLLM properly constructs URLs when using custom api_base
for dedicated endpoints.
"""

import os
import sys

import pytest

# Add the litellm package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


class TestVertexAIDedicatedEndpointSupport:
    """Test cases for Dedicated Endpoint URL construction"""

    def test_dedicated_endpoint_url_construction_basic(self):
        """Test basic dedicated endpoint URL construction"""
        vertex_base = VertexBase()
        dedicated_api_base = "https://my-dedicated-endpoint.com"
        endpoint_id = "1234567890"
        project_id = "test-project"
        location = "us-central1"
        use_dedicated_endpoint_format = True

        auth_header, url = vertex_base._check_custom_proxy(
            api_base=dedicated_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint_id,
            stream=False,
            auth_header="test-token",
            url="",
            model="any-model",
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1",
            use_dedicated_endpoint_format=use_dedicated_endpoint_format,
        )

        expected_url = f"{dedicated_api_base}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        assert url == expected_url, f"Expected {expected_url}, but got {url}"

    def test_dedicated_endpoint_url_construction_v1beta1(self):
        """Test dedicated endpoint URL construction with v1beta1 API version"""
        vertex_base = VertexBase()
        dedicated_api_base = "https://my-dedicated-endpoint.com"
        endpoint_id = "1234567890"
        project_id = "test-project"
        location = "us-central1"
        use_dedicated_endpoint_format = True

        auth_header, url = vertex_base._check_custom_proxy(
            api_base=dedicated_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint_id,
            stream=False,
            auth_header="test-token",
            url="",
            model="any-model",
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1beta1",
            use_dedicated_endpoint_format=use_dedicated_endpoint_format,
        )

        expected_url = f"{dedicated_api_base}/v1beta1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        assert url == expected_url, f"Expected {expected_url}, but got {url}"

    def test_dedicated_endpoint_with_trailing_slash(self):
        """Test that trailing slashes in api_base are handled correctly for dedicated endpoints"""
        vertex_base = VertexBase()
        dedicated_api_base = "https://my-dedicated-endpoint.com/"
        endpoint_id = "1234567890"
        project_id = "test-project"
        location = "us-central1"
        use_dedicated_endpoint_format = True

        auth_header, url = vertex_base._check_custom_proxy(
            api_base=dedicated_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint_id,
            stream=False,
            auth_header="test-token",
            url="",
            model="any-model",
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1",
            use_dedicated_endpoint_format=use_dedicated_endpoint_format,
        )

        expected_url = f"{dedicated_api_base.rstrip('/')}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        assert url == expected_url, f"Expected {expected_url}, but got {url}"

    def test_dedicated_endpoint_missing_params(self):
        """Test that missing required parameters raise ValueError for dedicated endpoints"""
        vertex_base = VertexBase()
        dedicated_api_base = "https://my-dedicated-endpoint.com"
        use_dedicated_endpoint_format = True

        # Missing vertex_project
        with pytest.raises(
            ValueError,
            match="vertex_project, vertex_location, and endpoint are required when use_dedicated_endpoint_format=True",
        ):
            vertex_base._check_custom_proxy(
                api_base=dedicated_api_base,
                custom_llm_provider="vertex_ai",
                gemini_api_key=None,
                endpoint="12345",
                stream=False,
                auth_header="test-token",
                url="",
                vertex_location="us-central1",
                vertex_project=None,
                use_dedicated_endpoint_format=use_dedicated_endpoint_format,
            )

        # Missing vertex_location
        with pytest.raises(
            ValueError,
            match="vertex_project, vertex_location, and endpoint are required when use_dedicated_endpoint_format=True",
        ):
            vertex_base._check_custom_proxy(
                api_base=dedicated_api_base,
                custom_llm_provider="vertex_ai",
                gemini_api_key=None,
                stream=False,
                auth_header="test-token",
                url="",
                endpoint="12345",
                vertex_location=None,
                vertex_project="test-project",
                use_dedicated_endpoint_format=use_dedicated_endpoint_format,
            )

        # Missing endpoint
        with pytest.raises(
            ValueError,
            match="vertex_project, vertex_location, and endpoint are required when use_dedicated_endpoint_format=True",
        ):
            vertex_base._check_custom_proxy(
                api_base=dedicated_api_base,
                custom_llm_provider="vertex_ai",
                gemini_api_key=None,
                stream=False,
                auth_header="test-token",
                url="",
                endpoint="",
                vertex_location="us-central1",
                vertex_project="test-project",
                use_dedicated_endpoint_format=use_dedicated_endpoint_format,
            )

    def test_dedicated_and_psc_mutually_exclusive(self):
        """Test that use_psc_endpoint_format and use_dedicated_endpoint_format are mutually exclusive"""
        vertex_base = VertexBase()
        dedicated_api_base = "https://my-dedicated-endpoint.com"

        with pytest.raises(
            ValueError,
            match="use_psc_endpoint_format and use_dedicated_endpoint_format are mutually exclusive, both cannot be True",
        ):
            vertex_base._check_custom_proxy(
                api_base=dedicated_api_base,
                custom_llm_provider="vertex_ai",
                gemini_api_key=None,
                stream=False,
                endpoint="",
                auth_header="test-token",
                url="",
                use_psc_endpoint_format=True,
                use_dedicated_endpoint_format=True,
            )
