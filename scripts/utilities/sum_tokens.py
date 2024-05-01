# Summing up the tokens from the given table
tokens_by_organization = [
    21418,    # United Nations
    156854,   # European Commission
    83968,    # Sustainable Development Solutions Networks
    191224,   # Finnish Government
    230323,   # German Government
    8254,     # Annual Review of Environment and Resources
    3743,     # Sustainability journal
    11554,    # Corporate Ownership and Control
    52030,    # Technology and Sustainable Development book
    7968,     # Section of Business Strategy for Sustainable Development
    16473     # Other sources
]

total_tokens = sum(tokens_by_organization)
print(total_tokens)