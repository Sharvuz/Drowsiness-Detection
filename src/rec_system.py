import pandas as pd

def recommend(gdf, emotion):
  for entry in gdf:
    if emotion=='happy' or emotion=='drowsy':
      return gdf.get_group(('high', 'high'))
    elif emotion=='neutral':
      return gdf.get_group(('high', 'low'))
    elif emotion=="angry" or emotion=="disgust" or emotion=="surprised":
      grp1 = gdf.get_group(('low', 'high'))
      grp2 = gdf.get_group(('high', 'low'))
      return grp1.append(grp2)
    elif emotion=="sad" or emotion=="scared":
      grp1 = gdf.get_group(('low', 'low'))
      grp2 = gdf.get_group(('high', 'high'))
      return grp1.append(grp2)

