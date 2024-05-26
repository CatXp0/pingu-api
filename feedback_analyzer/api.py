# importuri necesare pentru FastAPI, securitate, manipularea JWT,
# criptare si definirea modelelor de date
from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from .classifier.model import Model, get_model
from .config import Settings

# inițializarea aplicației FastAPI
app = FastAPI()

# funcție pentru obținerea configurației cu caching
@lru_cache
def get_settings():
    return Settings()

# algoritmul pentru criptarea JWT și durata de expirare a tokenului
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# Baza de date falsă pentru utilizatori
fake_users_db = {
    "pingu": {
        "username": "pingu",
        "hashed_password": "$2y$10$J2Ll8RC4mIKtpVJzxMxhM.QIpa9wuPYaoo/B1acXTH.JvDV63N.J2",
        "disabled": False,
    },
}
# contextul pentru criptarea parolelor
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# schema de autentificare OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# modele de date pentru Pydantic pentru gestionarea cererilor si raspunsurilor.
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


class ApiRequest(BaseModel):
    text: str


class ApiResponse(BaseModel):
    probabilities: Dict[str, float]
    affectiveItemsAnalysisRating: str
    confidence: float

# funcție pentru verificarea parolei, verifica daca parola introdusa corespunde cu hash-ul stocat
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# funcție pentru crearea hash-ului unei parole
def get_password_hash(password):
    return pwd_context.hash(password)

# funcție pentru obtinerea unui utilizator din baza de date falsa
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

# funcție pentru autentificarea utilizatorului verificand existenta si parola acestuia
def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# funcție pentru crearea unui token de acces JWT cu o durata de expirare
def create_access_token(app_secret_key: str, data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, app_secret_key, algorithm=ALGORITHM)
    return encoded_jwt

# funcție pentru obținerea utilizatorului curent pe baza token-ului JTW, care este decodificat
async def get_current_user(settings: Annotated[Settings, Depends(get_settings)],
                           token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # se decodifica tokenul primit cu cheia secreta
        payload = jwt.decode(token, settings.app_secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# funcție pentru obținerea utilizatorului curent activ
async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# endpoint pentru obținerea token-ului de acces
@app.post("/api/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
                settings: Annotated[Settings, Depends(get_settings)]):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        settings.app_secret_key, data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# endpoint pentru analizarea feedback-urilor
@app.post("/api/feedback/analyze", response_model=ApiResponse)
def predict(request: ApiRequest, token: Annotated[str, Depends(oauth2_scheme)],
            model: Model = Depends(get_model)):
    affective_items_analysis_rating, confidence, probabilities = model.predict(request.text)

    return ApiResponse(
        affectiveItemsAnalysisRating=affective_items_analysis_rating, confidence=confidence, probabilities=probabilities
    )
