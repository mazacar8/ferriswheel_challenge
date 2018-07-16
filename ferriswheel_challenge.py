import json, sys, argparse, urllib.request, gensim, re, os
import numpy as np
from scipy import spatial

def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser(\
		description = "Argument Parser for Product Classifier")
	parser.add_argument('data_filename', type=str,
						help="Path to json product data file")
	parser.add_argument('categories_filename', type=str,
						help="Path to txt file containing list of categories")
	parser.add_argument('categories_synonyms', type=str,\
		help="Path to txt file containing list of synonyms for categories")
	parser.add_argument('--stop',dest='stopwords_filename',type=str,\
		default=None,help="Optional path to txt file containing stopwords")
	parser.add_argument('--images',dest='write_images',type=int,default=0,\
		help="0 or 1. Option to write images of items into assigned category folders")

	return parser.parse_args()

def parse_files(data_f,cat_f,stop_f,syn_f,write_images=False):

	stopwords = []
	with open(data_f) as f:
		data = json.load(f)

	with open(cat_f) as f:
		categories = f.read().lower().splitlines()

	if stop_f:
		with open(stop_f) as f:
			stopwords = f.read().splitlines()

	cat_synonyms = dict()
	with open(syn_f) as f:
		for line in f.readlines():
			synonyms = line.strip().lower().split(' ')
			cat_synonyms[synonyms[0]] = synonyms

	#Creating/Emptying folders to write images into for each category
	if write_images:
		for cat in categories:
			if not os.path.exists(cat):
				os.makedirs(cat)
			else:
				for f in os.listdir(cat):
					os.unlink(os.path.join(cat,f))



	return data, categories, set(stopwords), cat_synonyms

def getWords(text):
	#RegEx to extract all words from description of item
	return re.compile('[a-z]+').findall(text)


def write_image(image_url,category,f_name):
	#Create image of item in appropriate category folder if possible
	f_path = os.path.join(category,f_name)
	if image_url[0:2] == '//':
		image_url = 'http:'+image_url

	try:
		with urllib.request.urlopen(image_url) as url_f, \
		open(f_path,'wb') as new_f:
			new_f.write(url_f.read())
	except:
		pass

def print_category_counts(categories,counts):
	print("\n\n\nFinal Category Counts:\n")
	for i in range(len(categories)):
		print("%s: %d"%(categories[i],counts[i]))


def categorize(data,cat_vecs,embeddings,stopwords,categories,\
	write_images=False):
#Categorize items based on their descriptions by doing similarity checks
#between embeddings of category synonyms and words in the description 

	threshold = 0.6
	counts = [0]*len(categories)

	for i,item in enumerate(data):

		description = item['description'].lower()
		words = getWords(description)
		category_found = False
		best_similarity = 0.0
		for word in words:
			if word not in stopwords and word in embeddings:
				for j in range(len(categories)-1):
					category = categories[j]
					for cat_vec in cat_vecs[category]:
						#Find similarity between word and category synonym
						#using embedding
						similarity = 1 - spatial.distance.cosine(cat_vec,\
							embeddings[word])

						#Keep track of most likely category
						if similarity > threshold and \
						similarity > best_similarity:
							best_similarity = similarity
							best_cat_index = j
							category_found = True
							f_name = "%d-Confidence-%.2f-with-%s"\
							%(i+1,similarity,word)

		#If no category seems likely, categorize into 'others'
		if not category_found:
			category = 'others'
			counts[-1] += 1
			f_name = "%d"%(i+1)

		else:
			category = categories[best_cat_index]
			counts[best_cat_index] += 1
			
		if write_images:
			write_image(item['images_url'],category,f_name)

		item['category'] = category

	with open('categorized_products.json', 'w') as outfile:
		json.dump(data, outfile, indent=0)

	print_category_counts(categories,counts)


def main(args):

	args = parse_arguments()

	print("Parsing Files...")
	data, categories, stopwords, cat_synonyms = \
	parse_files(args.data_filename,args.categories_filename,\
		args.stopwords_filename,args.categories_synonyms,args.write_images)

	embeddings = []
	cat_vecs = []
	print("Loading Word2Vec Model...")
	embeddings = gensim.models.KeyedVectors.load_word2vec_format\
	('./w2v/GoogleNews-vectors-negative300.bin', binary=True)

	print("Loading Category Embeddings...")
	cat_vecs = dict()
	#Load embeddings of all category synonyms
	for i in range(len(categories)-1):
		cat = categories[i]
		cat_vecs[cat] = []
		for syn in cat_synonyms[cat]:
			cat_vecs[cat].append(embeddings[syn])
	
	print("Categorizing items...")
	categorize(data,cat_vecs,embeddings,stopwords,categories,args.write_images)

if __name__ == "__main__":
	main(sys.argv)