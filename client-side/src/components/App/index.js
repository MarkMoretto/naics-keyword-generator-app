import React from "react"
import { Route, Switch } from "react-router-dom"

import Navbar from "../Navbar"
import About from "../pages/About"
import PageNotFound from "../pages/PageNotFound"

import "./styles.css"


const App = () => {
	return (
		<React.Fragment>
			<Navbar />
			<Switch>
				<Route exact path="/">
					<div className="container">
						<p>Hello!</p>
					</div>
				</Route>
				<Route path="/about">
					<About />
				</Route>
				<Route path="*">
					<PageNotFound />
				</Route>
			</Switch>
		</React.Fragment>
	);
}

export default App;
