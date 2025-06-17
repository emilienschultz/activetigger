import { ChangeEvent, FC, useEffect, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useGetGenModels } from '../../core/api';
import { GenerationModelApi, GenModel, SupportedAPI } from '../../types';

type FormValues = { model: string; name?: string; endpoint?: string; credentials?: string };

export const GenModelSetupForm: FC<{
  add: (model: Omit<GenModel & { api: SupportedAPI }, 'id'>) => void;
  cancel: () => void;
}> = ({ add, cancel }) => {
  const [availableAPIs, setAvailableAPIs] = useState<GenerationModelApi[]>([]);
  const [selectedAPI, setSelectedAPI] = useState<GenerationModelApi>(availableAPIs[0]);
  const [modelName, setModelName] = useState<string>('');
  const { models } = useGetGenModels();
  const { register, handleSubmit } = useForm<FormValues>();
  useEffect(() => {
    const fetchModels = async () => {
      setAvailableAPIs(await models());
    };
    fetchModels();
  }, [models]);
  useEffect(() => {
    setSelectedAPI(availableAPIs[0]);
  }, [availableAPIs]);

  const onAPIChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const index = parseInt(e.target.value);
    if (index >= availableAPIs.length) {
      throw new Error(`Invalid index choice: ${index}`);
    }
    setSelectedAPI(availableAPIs[index]);
  };

  const onSubmit: SubmitHandler<FormValues> = (data: FormValues) => {
    const slug = data.model;
    const name = modelName;
    const endpoint = data.endpoint;
    const credentials = data.credentials;
    if (slug === null) throw new Error('You should provide a model');
    add({
      slug,
      name,
      api: selectedAPI.name as SupportedAPI,
      endpoint,
      credentials,
    });
  };

  const onNameChange = (e: ChangeEvent<HTMLInputElement>) => {
    setModelName(e.target.value);
  };

  const onModelChange = (e: ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    setModelName(selectedAPI.name + '-' + e.target.value);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="d-flex flex-wrap gap-3 mb-3">
        <div className="form-floating col-5">
          <select
            className="form-select"
            id="api"
            defaultValue={0}
            onChange={onAPIChange}
            name="api"
          >
            {availableAPIs.map((api, index) => (
              <option key={index} value={index}>
                {api.name}
              </option>
            ))}
          </select>
          <label htmlFor="api">API </label>
        </div>
        {(() => {
          const inputs = [];
          if (selectedAPI !== undefined) {
            if (selectedAPI.name === 'OpenAI')
              inputs.push(
                <div className="form-floating col-5" key="model-select">
                  <select
                    className="form-control"
                    id="model"
                    {...register('model', { onChange: onModelChange })}
                  >
                    {selectedAPI.models.map((model) => (
                      <option key={model.slug} value={model.slug}>
                        {model.name}
                      </option>
                    ))}
                  </select>
                  {/* HACK: mt-0 is needed because a rule in the CSS file add a margin to .form-label. Maybe it should be removed*/}
                  <label htmlFor="model" className="mt-0">
                    Model
                  </label>
                </div>,
              );
            else {
              inputs.push(
                <div className="form-floating col-5" key="model-text">
                  <input
                    type="text"
                    id="model"
                    className="form-control"
                    placeholder="ID of the model"
                    {...register('model', { onChange: onModelChange })}
                  />
                  <label htmlFor="model" className="mt-0">
                    Model
                  </label>
                </div>,
              );
              if (selectedAPI.name !== 'OpenRouter')
                inputs.push(
                  <div className="form-floating col-5" key="endpoint">
                    <input
                      type="text"
                      id="endpoint"
                      className="form-control"
                      placeholder="enter the url of the endpoint"
                      {...register('endpoint')}
                    />
                    <label htmlFor="endpoint" className="mt-0">
                      Endpoint
                    </label>
                  </div>,
                );
            }

            if (selectedAPI.name !== 'Ollama')
              inputs.push(
                <div className="form-floating col-5" key="credentials">
                  <input
                    type="text"
                    id="credentials"
                    className="form-control"
                    placeholder="API key"
                    autoComplete="off"
                    {...register('credentials')}
                  />
                  <label htmlFor="credentials" className="mt-0">
                    API Credentials
                  </label>
                </div>,
              );
          }
          return inputs;
        })()}
        <div className="form-floating col-5">
          <input
            type="text"
            className="form-control"
            id="name"
            value={modelName}
            {...register('name', { onChange: onNameChange })}
          />
          <label htmlFor="name" className="mt-0">
            Name
          </label>
        </div>
      </div>
      <button type="submit" className="btn btn-primary me-1 col-1">
        Add
      </button>
      <button type="reset" value="Cancel" className="btn btn-outline-danger col-1" onClick={cancel}>
        Cancel
      </button>
    </form>
  );
};
