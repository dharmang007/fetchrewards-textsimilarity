# fetchrewards-textsimilarity
##  IMPORTANT: All the core code for finding the similarity is in textsimilarity.py and app.py (in getSimilarity()). You can also directly use the Commented Debugging code given at the bottom on textsimilarity.py file.

## Steps to run the project from Docker Hub
1) Install Docker on your machine
2) Run the command following command to pull the image and automated build from Docker hub
   > docker run index.docker.io/dharmang007/fetchrewards-textsimilarity:lastest
3) Go to http://172.17.0.2:5000/ 
4) Copy-paste the sample texts. Make sure you remove the extra white-space after copying the text. This will change the value of similarity.

## Steps to run the project from GitHub
1) Clone this repo.
2) Install the docker on your machine
3) Run command:
   > docker build --tag fetchrewards .
   > docker run fetchrewards

#Sample Text Result 

Sample 1
The easiest way to earn points with Fetch Rewards is to just shop for the products you already love. If you have
any participating brands on your receipt, you'll get points based on the cost of the products. You don't need to
clip any coupons or scan individual barcodes. Just scan each grocery receipt after you shop and we'll find the
savings for you.

Sample 2
The easiest way to earn points with Fetch Rewards is to just shop for the items you already buy. If you have any
eligible brands on your receipt, you will get points based on the total cost of the products. You do not need to cut
out any coupons or scan individual UPCs. Just scan your receipt after you check out and we will find the savings
for you.

Sample 3
We are always looking for opportunities for you to earn more points, which is why we also give you a selection
of Special Offers. These Special Offers are opportunities to earn bonus points on top of the regular points you
earn every time you purchase a participating brand. No need to pre-select these offers, we'll give you the points
whether or not you knew about the offer. We just think it is easier that way.

# Results
Cosine Similarity Between:
1) Sample 1 and Sample 2 : 0.73589231 (Higher Similarity)
   ![](sample1_2.png)
2) Sample 1 and Sample 3 : 0.184891731 (Less Similarity)
   ![](Sample1_3.png)

