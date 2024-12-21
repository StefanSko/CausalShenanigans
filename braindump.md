- include a more realistic scenario where u and v are not completely independent
- reserach searching vs. matching vs. ranking
- what about re-ranking?
- what about learning to rank?
- how can all of this be formalized in a DAG?


Story to tell
- consider a hypothetical restaurant review platform "TasteMap" matching users to restaurants
- users enter a query into a search bar indicating their cuisine preferences and their prefered location of the restaurant like, e.g. "Italian, in the city center"
- TasteMap then matches users to restaurants based on their preferences
- TasteMap then ranks the restaurants based on their popularity and relevance to the user's query
- TasteMap then displays the restaurants to the user
- The Product Owner receives some complaints from users that while the cuisine preferences usually match, the location preferences
- the product owner tasks the data sciente team to analyze the search results especially with respect to the location preferences
- the data scientists counterinutiviely discover a poor fit between the location preferences of the users and the location of the restaurants
- clearly the location preferences are important
- together with the Product owner they decide to improve the matching algorithm by weighting locations higher in the ranking
- they put forward thi new matching and AB test the conversion rate of this newly improved location centric approach
- counterintuitively the conversion rate is lower than the original approach
- what has happened?
