---
layout: post
title: Super Cool Web Scraper !!!!!!!!!!
---
In this blog post, I’m going to make a super cool web scraper. This web scraper will be able to start with an IMDB title page, and gather all the productions that the cast have been featured in. Then, based on number of shared actors per production, this will allow us generate recommendations of movies or tv shows to watch based on an existing liked movie or tv show. Here’s how we set up the project…

First, find a TV show or Movie on IMDB and copy the url. Once you have this, we can begin writing our methods. First, the parse method...

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-29%20at%203.46.59%20PM.png?raw=true)

This method works by starting at the title page for Peaky Blinders. This method sends the spider from an IMDB title page, to the page for Cast and Crew. It yields a scrapy request that calls the next method, parse_full_credits, on the new page. 

Then we write the parse_full_credits method... 

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-29%20at%203.47.07%20PM.png?raw=true)

This method works by being called on an IMDB cast and
crew page, in this case the Peaky Blinders cast and crew page. This method first scans the page for the list of 
actors, then compiles the proper url phrases for each actor.
Then, with this list of url phrases, we add https://www.imdb.com/ to the front and yield a scrapy request for the resulting url string, calling the method, parse_actor_page, on the new page.

Then we ...

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-29%20at%203.47.36%20PM.png?raw=true)

This method first leads the scraper to look at elements in the filmography section and then I specified to only include elements with the Id tag starting with actor. I then scraped all the text without the in_production class and extracted to get all the movie titles. To get the actor name, I just looked at the actor overview and then scraped the text and extracted the first element to get the character name. To finish, I yield a dictionary with the actor name and movies they are in.

To demonstrate my results, I created a table demonstrating the shared productions of the actors and a bar graph showing the top 5 results.

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-04%20at%207.38.43%20PM.png?raw=true)

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-29%20at%203.46.16%20PM.png?raw=true)

Link to Repository: https://github.com/jameswest25/IMDB_scraper/tree/main/IMDB_scraper/IMDB_scraper