# coding: utf-8
import  pandas as pd
import numpy
import matplotlib.pyplot as plt
from  sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_table("D:\Data_Minig\seance4_analyse_descriptive(ACP)\entreprises.txt",sep ='\t',header = 0)
#print  df
df_clean = df.drop(['Ent','ET'],axis=1)
print df_clean

#centrage et réduction
sc = StandardScaler()
df_new = sc.fit_transform(df_clean)
print(df_new)
#ACP
acp = PCA(svd_solver='full')
print(acp)
coordonnes = acp.fit_transform(df_new)

print acp.n_components_
print acp.explained_variance_ratio_
#critère du coude => k=2
plt.plot([1,2,3,4],numpy.cumsum(acp.explained_variance_ratio_))
plt.title("Explained variance / number of factors")
plt.ylabel("variance")
plt.xlabel("dim")
#plt.show()
print   acp.components_

#cercle de corrélation
(fig, ax) = plt.subplots(figsize=(12, 12))
for i in range(0, len(acp.components_)):
    ax.arrow(0,0,  # commencer depuis l'origine
             acp.components_[0, i],
             acp.components_[1, i],
             head_width=0.1,
             head_length=0.1)

####### ajouter le cercle#########
an = numpy.linspace(0, 2 * numpy.pi, 100)
plt.plot(numpy.cos(an), numpy.sin(an))
##################################
plt.axis('equal')
plt.xlabel('dim1')
plt.ylabel('dim2')
ax.set_title('cercle de correlation')
plt.show()