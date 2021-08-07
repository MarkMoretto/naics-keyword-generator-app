import { useEffect, useRef, useState } from "react"
import Griddy from "../../Griddy"

import "./styles.css"


let baseUrl = new URL("http://127.0.0.1:8000/stopwords")

const inputPlaceholder = "10"


const Stopwords = () => {
    const [stopwordsUrl, setStopwordsUrl] = useState(baseUrl)
    const [stopwords, setStopwords] = useState({})
    const [qryParams, setQryParams] = useState({
        sample_size: "",
    })
    const refTextInput = useRef()

    const onChange = e => {
        setQryParams(prevState => {
            return {
                ...prevState,
                [e.target.name]: e.target.value,
            }
        })
    }

    const updateQuery = (el) => {
        let currentKey = Object.keys(el)[0];
        if (el.sample_size.length > 0) {
            baseUrl.searchParams.set(currentKey, el.sample_size)
        } else {
            baseUrl.searchParams.delete(currentKey)
        }
        setStopwordsUrl(baseUrl)
        console.log(`${stopwordsUrl}`)
        // console.log(stopwordsUrl)
    }


    useEffect(() => {
        // console.log("useEffect called.")
        updateQuery(qryParams)
        // refTextInput.current.focus()
    })

    const resetPage = () => {
        setStopwords({})
        setQryParams({
            sample_size: "",
        })
        refTextInput.current.focus()
    }

 
    const handleSubmit = e => {
        e.preventDefault()

        fetch(baseUrl)
        .then(response => response.json())
        .then((data) => {
            setStopwords(data)          
        })
        .catch(console.error)
    }

    return (
        <>
            <h2>Random English Stopwords!</h2>
            <h5>Enter the number of words to return and press Go!</h5>
            <div>
                <form onSubmit={handleSubmit} onReset={resetPage} className="form-container">
                    <label htmlFor="id_sample_size">No. of Words:</label>
                    <input 
                        id="id_sample_size"
                        name="sample_size"
                        type="text"
                        placeholder={inputPlaceholder}
                        value={qryParams.sample_size}
                        onChange={onChange}
                        ref={refTextInput}
                        onFocus={(e) => e.target.placeholder = ""} 
                        onBlur={(e) => e.target.placeholder = `${inputPlaceholder}`}                   
                    />
                    <button class="stopword-btn" type="submit">Go!</button>
                    <button class="stopword-btn" type="reset">Reset</button>
                </form>
            </div>

            <div className="stopword-list">
                <Griddy itemArray={stopwords} numberColumns={4} classname={"stopword-list"} />
            </div>     
        </>
    )
}

export default Stopwords