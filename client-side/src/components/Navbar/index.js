import { NavLink } from "react-router-dom"
import navbarLinks from "./nav-links"

import BrandLogo from "../BrandLogo"

import "./styles.css"

const Navbar = () => {
    return (
        <nav className="navBar">
                <BrandLogo />
                <ul>
                    {navbarLinks.map(item => {
                        return (
                        <li key={item.id}>
                            <NavLink to={item.path} activeClassName="active-link" exact>
                                {item.text}
                            </NavLink>
                        </li>
                        )
                    })}
                </ul>
        </nav>
    )
}

export default Navbar