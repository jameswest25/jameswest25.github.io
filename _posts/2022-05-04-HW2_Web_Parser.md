---
layout: post
title: HW2 Web Scraping
---
In this blog post, I’m going to make a super cool web scraper… Here’s a link to my project repository… Here’s how we set up the project…

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-04%20at%207.40.26%20PM.png?raw=true)

This method works by starting at the title page for Peaky Blinders. It then takes advantage of the phrase 'fullcredits' to send the parser to the Cast and Crew page without having to scrape the entire page.

Then we ... 

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-04%20at%207.40.33%20PM.png?raw=true)

In this method, we scanned the list of actors and extracted their links. However, these links consisted of only the ending, so we had to add the imdb link to them. Then, we used these links for the third method...

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-04%20at%207.40.55%20PM.png?raw=true)

For this method, I first lead to the scraper to look at elements with the class filmo-row and then I specified to only include elements with the Id tag starting with actor. I then scraped all the text without the in_production class and extracted to get all the movie titles. To get the actor name, I just looked at the element with class name_overview.with-hero and then scraped text and extracted the first element to get the character name. To finish, I yield multiple dictionaries to fill out the csv. 

To demonstrate my results, I created a table demonstrating the shared productions of the actors and a bar graph showing the top 5 results.

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-04%20at%207.38.43%20PM.png?raw=true)

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-04%20at%207.38.35%20PM.png?raw=true)

Link to Repository: https://github.com/jameswest25/IMDB_scraper/tree/main/IMDB_scraper/IMDB_scraper