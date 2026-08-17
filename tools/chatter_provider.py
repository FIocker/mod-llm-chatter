"""Shared provider-client construction."""


def create_anthropic_client(anthropic_module, config, api_key=None):
    """Create an Anthropic client, optionally against a compatible endpoint."""
    key = api_key or config.get('LLMChatter.Anthropic.ApiKey', '')
    kwargs = {'api_key': key}
    base_url = config.get('LLMChatter.Anthropic.BaseUrl', '').strip()
    if base_url:
        kwargs['base_url'] = base_url.rstrip('/') + '/'
    return anthropic_module.Anthropic(**kwargs)
