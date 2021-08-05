import React from "react"
import { Route, Switch } from "react-router-dom"

import Navbar from "../Navbar"
import About from "../pages/About"
import KeywordSimilarity from "../pages/KeywordSimilarity"
import Stopwords from "../pages/Stopwords"
import PageNotFound from "../pages/PageNotFound"


const App = () => {
	return (
		<React.Fragment>
			<Navbar />
			<div className="outer-container">
				<Switch>		
					<Route exact path="/">
						<div className="inner-container">
							<h1>Hello!</h1>
						</div>
					</Route>
					<Route path="/keyword-similarity">
						<div className="inner-container">
							<KeywordSimilarity />
						</div>
					</Route>
					<Route path="/random-stopwords">
						<div className="inner-container">
							<Stopwords />
						</div>
					</Route>					
					<Route path="/about">
						<About />
					</Route>			
					<Route path="*">
						<PageNotFound />
					</Route>
				</Switch>
			</div>			
		</React.Fragment>
	);
}

export default App;
