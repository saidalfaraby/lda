import lda
import lda.utils
import string,re

v,l,ws,ds = lda.utils.files_to_lists('clean_dataset')

x=lda.utils.lists_to_matrix(ws,ds)

model = lda.LDA(n_topics=2, n_iter=20, random_state=1)
model.fit(x)

#print t
#print set(t)