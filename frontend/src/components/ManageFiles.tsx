import { FC, useState } from 'react';
import { useUploadData } from '../core/api';

interface FileUploadProps {}

// to do :
// get available files
// delete existing file

export const ManageFiles: FC<FileUploadProps> = () => {
  const [file, setFile] = useState<File | null>(null);
  const uploadData = useUploadData();

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    uploadData(file);
  };

  const formData = new FormData();
  formData.append('file', file || '');

  return (
    <div>
      <div className="explanations">Upload tabular file to use</div>
      <div className="d-flex align-items-center">
        <input type="file" onChange={handleFileChange} className="form-control" />
        <button onClick={handleUpload} disabled={!file} className="btn btn-primary">
          Upload
        </button>
      </div>
    </div>
  );
};
