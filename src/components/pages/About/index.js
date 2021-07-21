import { Link, Route, useRouteMatch } from "react-router-dom"
import RouterPage from "../RouterPage"

const About = () => {

    const { url, path} = useRouteMatch()

    return (
        <div>
            <ul>
                <li>
                    <Link to={`${url}/about-app`}>About App</Link>
                </li>
                <li>
                    <Link to={`${url}/about-author`}>About Author</Link>
                </li>
                <li>
                    <Route path={`${path}/:sluggish`}>
                        <RouterPage />
                    </Route>
                </li>
            </ul>
        </div>
    )
}

export default About