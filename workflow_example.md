# workflow

s: H = "taylor swift is 21 years old and moved to nashvile and she likes the colour red"
t_src: [(s,p,o)] = [
  (taylor swift, age, 21),
  (taylor swift, moved, nashvile),
  (taylor swift, likes, red)
]
t_dist: [(s, p, o')] = [
  (taylor swift, age, 31),
  (taylor swift, moved, nashvile),
  (taylor swift, likes, red)
]
verfied = [false, true, true]
s': corrected H = "taylor swift is 31 years old and moved to nashvile and she likes the colour red"

# arch

H -> extractor (.) -> t_src
t_src, KG -> retriever (.,.) -> t_dist
t_dist -> verifier (.) -> verfied
t_dist -> constructor (.) -> s'

s.t. constructor = extract^-1

# assumptions (not our problem today)

if the LLM provides a sentence (s: H) that is not in the Knowledge Graph (KG), then the extractor will not be able to extract any triples from it
t_src = t_dist if and only if verfied is true


# adjust model , extended triplets , spacy 