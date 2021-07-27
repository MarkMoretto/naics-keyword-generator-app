
// Grid helper
// TODO: Possibly add prop for number of columns.
import styles from "./Griddy.module.css"

const Griddy = ({ itemArray, numberColumns = 4, classname = ""}) => {
    
    const colStyle = `col-${numberColumns}`

    return (
        <div className={classname}>
            <ul className={`${styles["grid-list"]} ${numberColumns ? styles[colStyle] : ""}`}>
            {Object.entries(itemArray).length > 0 
            && Object.values(itemArray)[0].map(item => {
                return (
                    <li id={item}>{item}</li>
                    )
                })
            }
            </ul>
        </div>          
    )
}

export default Griddy
