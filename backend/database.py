from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DatasetInfo(Base):
    __tablename__ = "dataset_info"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    filename = Column(String(200), nullable=False)
    records = Column(Integer, default=0)
    features = Column(Integer, default=0)
    severity_classes = Column(Integer, default=0)
    class_distribution = Column(Text, default="{}")
    date_range_start = Column(String(50), default="")
    date_range_end = Column(String(50), default="")
    status = Column(String(50), default="not_loaded")
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    dataset_name = Column(String(200), nullable=False)
    accuracy = Column(Float, default=0.0)
    precision_weighted = Column(Float, default=0.0)
    recall_weighted = Column(Float, default=0.0)
    f1_weighted = Column(Float, default=0.0)
    f1_macro = Column(Float, default=0.0)
    roc_auc = Column(Float, default=0.0)
    cohens_kappa = Column(Float, default=0.0)
    mcc = Column(Float, default=0.0)
    log_loss_value = Column(Float, default=0.0)
    cv_mean = Column(Float, default=0.0)
    cv_std = Column(Float, default=0.0)
    training_time = Column(Float, default=0.0)
    confusion_matrix_json = Column(Text, default="[]")
    normalized_cm_json = Column(Text, default="[]")
    roc_data_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    input_data = Column(Text, nullable=False)
    prediction = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.0)
    probabilities = Column(Text, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)


class EDAResult(Base):
    __tablename__ = "eda_results"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(200), nullable=False)
    chart_name = Column(String(100), nullable=False)
    chart_type = Column(String(50), nullable=False)
    chart_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ShapResult(Base):
    __tablename__ = "shap_results"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    dataset_name = Column(String(200), nullable=False)
    feature_importance_json = Column(Text, default="[]")
    summary_plot_base64 = Column(Text, default="")
    bar_plot_base64 = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    create_tables()
    print("All database tables created successfully!")
