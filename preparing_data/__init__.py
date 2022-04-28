# prepare and pre-process crawling_data and construct peersum dataset

# structure of prepared peersum for ICLR 2017
# paper_id,
# title,
# abstract,
# score,
# acceptance,
# meta_review,
# reviews, [writer, content (rating, confidence, comment)]
# label

# structure of prepared peersum for ICLR 2018-2022, ICLR 2020 has no confidence
# paper_id,
# title,
# abstract,
# acceptance,
# meta_review,
# reviews, [review_id, writer, content (rating, confidence, comment), replyto]   review_id and replyto are for the conversation structure
# label

# structure of prepared peersum for NIPS 2019-2020
# paper_id,
# title,
# abstract,
# acceptance,
# meta_review,
# reviews, [writer, content (comment)]
# label

# structure of prepared peersum for NIPS 2021
# paper_id,
# title,
# abstract,
# acceptance,
# meta_review,
# reviews, [review_id, writer, content (rating, confidence, comment), replyto]   review_id and replyto are for the conversation structure
# label


# label: train, val, test
# writer: official_reviewer, public, author