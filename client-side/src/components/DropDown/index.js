
import { useEffect,  useRef} from "react"


import styles from "./DropDown.module.css"

// https://learnersbucket.com/examples/react/create-dropdown-in-react/

const DropDown = ({ label, isDropDownOpen, selectOptions }, props) => {
    const refDropDown = useRef()
    const refDisplayArea = useRef()

    useEffect(() => {
        console.log("DropDown useEffect() called.")
    })

    return (
        <div className={styles.wrapper}>
            <div
                className={styles.dropToggler}
                onClick={this.toggleDropDown}
                ref={refDropDown}
            >
                <span className={styles.label}>{label}</span>
                <span className={styles.arrow}>{isDropDownOpen ? "\u25B2" : "\u25BC"}</span>
            </div>
            <div className={styles.displayArea}>
            {isDropDownOpen && (
                <div
                    className={styles.children}
                    ref={refDisplayArea}
                >
                {selectOptions}
                </div>
            )}
            </div>
        </div>
    )
}



export default DropDown