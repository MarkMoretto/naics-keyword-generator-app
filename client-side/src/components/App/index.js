import React from "react"
import { Route, Switch } from "react-router-dom"

import Navbar from "../Navbar"
import About from "../pages/About"
import Stopwords from "../pages/Stopwords"
import PageNotFound from "../pages/PageNotFound"

// client-side\src\static\image\brands\humana\h-logo.svg
// import HLogo from "../../static/image/brands/humana/h-logo.svg"
// import Humana from "../logos/Humana"

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
